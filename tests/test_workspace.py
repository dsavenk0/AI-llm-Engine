"""Security and end-user flow for Google Workspace OAuth without contacting Google or Keychain."""
import json
import os
import tempfile
import unittest
import urllib.parse
from unittest import mock

from assistant import workspace


CLIENT = "12345-example.apps.googleusercontent.com"


class FakeOAuthServer:
    def __init__(self, address, handler):
        self.server_address = ("127.0.0.1", 43123)
        self.closed = False

    def handle_request(self):
        return None

    def server_close(self):
        self.closed = True


class WorkspaceOAuthTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = os.path.join(self.tmp.name, "home")
        self.root = os.path.join(self.tmp.name, "root")
        os.makedirs(os.path.join(self.root, "assistant"))

    def test_bundled_client_is_one_click_and_status_never_exposes_credentials(self):
        with open(os.path.join(self.root, "assistant", "google-oauth-client-id.txt"), "w") as f:
            f.write(CLIENT)
        w = workspace.Workspace(self.home, self.root)
        with mock.patch.object(w, "_keychain_get", return_value={"refresh_token": "secret", "access_token": "also-secret",
                                                                  "scopes": [workspace.SCOPES["sheets"]]}):
            status = w.status()
        self.assertEqual(status, {"configured": True, "client": "bundled", "connected": True,
                                  "scopes": [workspace.SCOPES["sheets"]]})
        self.assertNotIn("secret", json.dumps(status))

    def test_custom_client_is_owner_only_and_overrides_the_bundled_one(self):
        w = workspace.Workspace(self.home, self.root)
        with mock.patch.object(w, "_keychain_get", return_value=None):
            status = w.configure(CLIENT)
        self.assertEqual(status["client"], "custom")
        self.assertEqual(os.stat(w.config_path).st_mode & 0o777, 0o600)

    def test_authorize_opens_pkce_itself_and_does_not_return_state_or_url(self):
        w = workspace.Workspace(self.home, self.root)
        with mock.patch.dict(os.environ, {"LLMBAR_GOOGLE_CLIENT_ID": CLIENT}), \
             mock.patch.object(workspace.http.server, "ThreadingHTTPServer", FakeOAuthServer), \
             mock.patch.object(workspace.webbrowser, "open", return_value=True) as opened:
            result = w.authorize(["sheets", "calendar"])
        self.assertEqual(result, {"started": True, "scopes": ["sheets", "calendar"]})
        self.assertNotIn("url", result)
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(opened.call_args.args[0]).query)
        self.assertEqual(query["code_challenge_method"], ["S256"])
        self.assertEqual(query["redirect_uri"], ["http://127.0.0.1:43123/oauth2/callback"])
        self.assertIn(query["state"][0], w.pending)
        self.assertNotEqual(query["code_challenge"][0], w.pending[query["state"][0]]["verifier"])

    def test_bad_client_ids_are_rejected(self):
        w = workspace.Workspace(self.home, self.root)
        for value in ("", "secret", "https://example.com", "x" * 300 + ".apps.googleusercontent.com"):
            with self.assertRaises(workspace.WorkspaceError):
                w.configure(value)


class WorkspaceOperationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.w = workspace.Workspace(self.tmp.name)
        self.calls = []

    def api(self, method, url, scope, body=None, retry=True):
        self.calls.append((method, url, scope, body))
        if "sheets.googleapis.com" in url and method == "GET": return {"values": []}
        if "sheets.googleapis.com" in url and method == "PUT": return {"updatedRange": "Sheet1!B4"}
        if "docs.googleapis.com" in url and method == "GET": return {"body": {"content": [{"endIndex": 7}]}}
        if "drive/v3/files?" in url: return {"id": "document_12345", "webViewLink": "https://docs.google.com/x"}
        if "gmail.googleapis.com" in url and method == "POST": return {"id": "draft_12345"}
        if "calendar" in url and method == "POST": return {"id": "event_12345", "htmlLink": "https://calendar.google.com/x"}
        return {}

    def execute(self, action):
        with mock.patch.object(self.w, "_api", side_effect=self.api):
            return self.w.execute(action)

    def test_named_operations_use_only_the_expected_google_apis_and_return_private_undo_records(self):
        sheet = self.execute({"action": "workspace_sheets_write", "spreadsheet_id": "spreadsheet_12345",
                          "cell": "Sheet1!B4", "value": "hello", "overwrite": False})
        self.assertEqual(sheet["_undo"]["kind"], "sheets")
        self.assertEqual(self.calls[-1][3]["values"], [["hello"]])

        doc = self.execute({"action": "workspace_docs_append", "document_id": "document_12345", "text": "note"})
        self.assertEqual(doc["_undo"], {"kind": "docs", "document_id": "document_12345", "start": 6, "end": 10})

        drive = self.execute({"action": "workspace_drive_create", "name": "Plan", "text": "body"})
        self.assertEqual(drive["_undo"], {"kind": "drive", "file_id": "document_12345"})

        gmail = self.execute({"action": "workspace_gmail_draft", "to": "person@example.com", "subject": "Hi", "body": "Draft"})
        self.assertEqual(gmail["_undo"]["kind"], "gmail")
        self.assertIn("raw", self.calls[-1][3]["message"])

        event = self.execute({"action": "workspace_calendar_create", "summary": "Review",
                          "start": "2026-11-13T12:00:00+01:00", "end": "2026-11-13T12:30:00+01:00",
                          "time_zone": "Europe/Madrid"})
        self.assertEqual(event["_undo"]["kind"], "calendar")

    def test_sheets_refuses_an_unapproved_overwrite(self):
        with mock.patch.object(self.w, "_api", return_value={"values": [["existing"]]}):
            with self.assertRaisesRegex(workspace.WorkspaceError, "not empty"):
                self.w.execute({"action": "workspace_sheets_write", "spreadsheet_id": "spreadsheet_12345",
                                "cell": "A1", "value": "new", "overwrite": False})

    def test_gmail_draft_rejects_header_injection_before_an_api_call(self):
        actions = [
            {"action": "workspace_gmail_draft", "to": "person@example.com\nBcc: thief@example.com",
             "subject": "Hi", "body": "Draft"},
            {"action": "workspace_gmail_draft", "to": "person@example.com",
             "subject": "Hi\r\nBcc: thief@example.com", "body": "Draft"},
        ]
        for action in actions:
            with self.assertRaisesRegex(workspace.WorkspaceError, "line breaks"):
                self.execute(action)
        self.assertEqual(self.calls, [])

    def test_every_created_workspace_object_has_a_bounded_undo(self):
        records = [
            {"kind": "sheets", "spreadsheet_id": "spreadsheet_12345", "cell": "A1", "values": []},
            {"kind": "docs", "document_id": "document_12345", "start": 1, "end": 4},
            {"kind": "drive", "file_id": "document_12345"},
            {"kind": "gmail", "draft_id": "draft_12345"},
            {"kind": "calendar", "event_id": "event_12345"},
        ]
        with mock.patch.object(self.w, "_api", side_effect=self.api):
            for record in records: self.assertTrue(self.w.undo(record)["ok"])
        methods = [call[0] for call in self.calls]
        self.assertEqual(methods, ["POST", "POST", "PATCH", "DELETE", "DELETE"])

    def test_arbitrary_workspace_requests_are_never_accepted(self):
        for action in ({"action": "workspace_http", "url": "https://evil.test"}, {"action": "delete_everything"}):
            with self.assertRaisesRegex(workspace.WorkspaceError, "unsupported"):
                self.w.execute(action)


if __name__ == "__main__":
    unittest.main()
