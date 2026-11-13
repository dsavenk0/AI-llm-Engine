"""Google Workspace OAuth and a narrow local API.

Browser pages and browser extensions never receive an OAuth access or refresh token.  A desktop OAuth client id
is stored in the assistant's owner-only configuration; refresh tokens live in the macOS Keychain.  Authorization
uses PKCE and an in-memory, one-time loopback state value.
"""
import base64
import datetime
import email.message
import email.utils
import hashlib
import http.server
import json
import os
import re
import secrets
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import urllib.error
import webbrowser

AUTH = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN = "https://oauth2.googleapis.com/token"
KEYCHAIN_SERVICE = "local.llm-engine.workspace"
SCOPES = {
    "sheets": "https://www.googleapis.com/auth/spreadsheets",
    "docs": "https://www.googleapis.com/auth/documents",
    "drive": "https://www.googleapis.com/auth/drive.file",
    "gmail_read": "https://www.googleapis.com/auth/gmail.readonly",
    "gmail_compose": "https://www.googleapis.com/auth/gmail.compose",
    "calendar": "https://www.googleapis.com/auth/calendar.events",
}


class WorkspaceError(RuntimeError):
    pass


def _b64(data):
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


class Workspace:
    def __init__(self, home, root=None):
        self.home, self.root = home, root
        self.config_path = os.path.join(home, "workspace.json")
        self.pending = {}  # state -> {verifier, redirect, scopes, expires}
        self.lock = threading.Lock()
        self._status_cache, self._status_at = None, 0.0

    def _config(self):
        try:
            with open(self.config_path) as f: data = json.load(f)
        except (OSError, ValueError): data = {}
        return data if isinstance(data, dict) else {}

    def configure(self, client_id):
        client_id = str(client_id or "").strip()
        if not client_id.endswith(".apps.googleusercontent.com") or len(client_id) > 250:
            raise WorkspaceError("enter a Google Desktop OAuth client ID ending in .apps.googleusercontent.com")
        os.makedirs(self.home, mode=0o700, exist_ok=True)
        fd = os.open(self.config_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f: json.dump({"client_id": client_id}, f)
        os.chmod(self.config_path, 0o600)
        self._status_at = 0.0
        return self.status()

    def _keychain_get(self):
        try:
            r = subprocess.run(["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-a", "google", "-w"],
                               text=True, capture_output=True, timeout=5)
            return json.loads(r.stdout) if r.returncode == 0 else None
        except (OSError, ValueError, subprocess.SubprocessError): return None

    def _keychain_set(self, token):
        data = json.dumps(token, separators=(",", ":"))
        try:
            r = subprocess.run(["security", "add-generic-password", "-U", "-s", KEYCHAIN_SERVICE, "-a", "google", "-w", data],
                               text=True, capture_output=True, timeout=8)
        except OSError as e: raise WorkspaceError(f"could not save the Google session in Keychain: {e}") from None
        if r.returncode: raise WorkspaceError("could not save the Google session in Keychain")

    def disconnect(self):
        try:
            subprocess.run(["security", "delete-generic-password", "-s", KEYCHAIN_SERVICE, "-a", "google"],
                           capture_output=True, timeout=8)
        except (OSError, subprocess.SubprocessError):
            pass
        self._status_at = 0.0
        return self.status()

    def status(self):
        if self._status_cache is not None and time.time() - self._status_at < 5:
            return dict(self._status_cache)
        token, cfg = self._keychain_get(), self._config()
        client_id, source = self._client_id()
        self._status_cache = {"configured": bool(client_id), "client": source,
                              "connected": bool(token and token.get("refresh_token")),
                              "scopes": token.get("scopes", []) if isinstance(token, dict) else []}
        self._status_at = time.time()
        return dict(self._status_cache)

    def authorize(self, names):
        client_id, _ = self._client_id()
        if not client_id: raise WorkspaceError("configure a Google Desktop OAuth client ID first")
        wanted = [SCOPES[n] for n in names if n in SCOPES]
        if not wanted: raise WorkspaceError("choose at least one Workspace permission")
        # OAuth desktop apps redirect to a loopback listener.  The verifier is never written to disk.
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), self._callback_handler)
        server.timeout = 600  # abandoned sign-in must not leave a listener thread blocked forever
        redirect = f"http://127.0.0.1:{server.server_address[1]}/oauth2/callback"
        state, verifier = secrets.token_urlsafe(32), secrets.token_urlsafe(64)
        challenge = _b64(hashlib.sha256(verifier.encode()).digest())
        with self.lock: self.pending[state] = {"verifier": verifier, "redirect": redirect, "scopes": wanted, "expires": time.time() + 600}
        def receive_once():
            try: server.handle_request()
            finally: server.server_close()
        threading.Thread(target=receive_once, daemon=True).start()
        query = urllib.parse.urlencode({"client_id": client_id, "redirect_uri": redirect, "response_type": "code",
                                        "scope": " ".join(wanted), "state": state, "code_challenge": challenge,
                                        "code_challenge_method": "S256", "access_type": "offline", "prompt": "consent"})
        url = AUTH + "?" + query
        if not webbrowser.open(url):
            server.server_close()
            with self.lock: self.pending.pop(state, None)
            raise WorkspaceError("could not open the Google sign-in page")
        # Do not return the authorization URL or its one-time state to the extension.
        return {"started": True, "scopes": names}

    def _client_id(self):
        custom = self._config().get("client_id")
        if isinstance(custom, str) and custom.endswith(".apps.googleusercontent.com"):
            return custom, "custom"
        value = os.environ.get("LLMBAR_GOOGLE_CLIENT_ID", "").strip()
        bundled = os.path.join(self.root, "assistant", "google-oauth-client-id.txt") if self.root else ""
        if not value and bundled:
            try:
                with open(bundled) as f: value = f.read().strip()
            except OSError: pass
        return (value, "bundled") if value.endswith(".apps.googleusercontent.com") else (None, None)

    def _callback_handler(self, request, address, server):
        owner = self
        class Callback(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_): pass
            def do_GET(self):
                q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
                state, code = (q.get("state") or [""])[0], (q.get("code") or [""])[0]
                with owner.lock: pending = owner.pending.pop(state, None)
                ok, message = False, "Authorization was not accepted. Return to LLMBar."
                if pending and pending["expires"] > time.time() and code:
                    try: owner._exchange(code, pending); ok, message = True, "Google Workspace is connected. You can close this tab."
                    except WorkspaceError as e: message = f"Could not connect: {e}"
                body = ("<!doctype html><meta charset=utf-8><title>LLMBar</title><p>" +
                        html_escape(message) + "</p>").encode()
                self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        return Callback(request, address, server)

    def _exchange(self, code, pending):
        client_id, _ = self._client_id()
        if not client_id: raise WorkspaceError("OAuth client configuration disappeared")
        body = urllib.parse.urlencode({"code": code, "client_id": client_id, "redirect_uri": pending["redirect"],
                                       "grant_type": "authorization_code", "code_verifier": pending["verifier"]}).encode()
        try:
            with urllib.request.urlopen(urllib.request.Request(TOKEN, data=body, headers={"Content-Type": "application/x-www-form-urlencoded"}), timeout=15) as r:
                token = json.load(r)
        except Exception as e: raise WorkspaceError("Google rejected the authorization response") from e
        if not isinstance(token, dict) or not token.get("refresh_token"): raise WorkspaceError("Google did not return a refresh token")
        token["scopes"] = pending["scopes"]; token["saved_at"] = int(time.time()); self._keychain_set(token)
        self._status_at = 0.0

    def _access_token(self, scope, force=False):
        token = self._keychain_get()
        if not isinstance(token, dict) or not token.get("refresh_token"):
            raise WorkspaceError("connect Google Workspace first")
        if scope not in token.get("scopes", []):
            raise WorkspaceError("this Google permission was not granted; reconnect Workspace")
        expires = int(token.get("saved_at") or 0) + int(token.get("expires_in") or 0)
        if not force and token.get("access_token") and expires > time.time() + 60:
            return token["access_token"]
        client_id, _ = self._client_id()
        if not client_id: raise WorkspaceError("OAuth client configuration disappeared")
        body = urllib.parse.urlencode({"client_id": client_id, "refresh_token": token["refresh_token"],
                                       "grant_type": "refresh_token"}).encode()
        try:
            with urllib.request.urlopen(urllib.request.Request(TOKEN, data=body, headers={
                    "Content-Type": "application/x-www-form-urlencoded"}), timeout=15) as r:
                fresh = json.load(r)
        except Exception as e:
            raise WorkspaceError("Google session expired; reconnect Workspace") from e
        if not isinstance(fresh, dict) or not fresh.get("access_token"):
            raise WorkspaceError("Google did not refresh the session")
        saved = dict(token, **fresh, refresh_token=token["refresh_token"], scopes=token.get("scopes", []),
                     saved_at=int(time.time()))
        self._keychain_set(saved); self._status_at = 0.0
        return saved["access_token"]

    def _api(self, method, url, scope, body=None, retry=True):
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
        headers = {"Authorization": "Bearer " + self._access_token(scope), "Accept": "application/json"}
        if data is not None: headers["Content-Type"] = "application/json; charset=utf-8"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers, method=method),
                                        timeout=20) as r:
                raw = r.read(2_000_000)
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            if e.code == 401 and retry:
                self._access_token(scope, force=True)
                return self._api(method, url, scope, body, retry=False)
            try:
                detail = json.loads(e.read(64_000)).get("error", {}).get("message")
            except Exception: detail = None
            raise WorkspaceError(detail or f"Google API rejected the operation ({e.code})") from None
        except (OSError, ValueError) as e:
            raise WorkspaceError("could not reach Google Workspace") from e

    @staticmethod
    def _id(value, label):
        value = str(value or "")
        if not re.fullmatch(r"[A-Za-z0-9_-]{10,200}", value):
            raise WorkspaceError(f"invalid {label}")
        return value

    @staticmethod
    def _text(value, label, limit):
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            raise WorkspaceError(f"{label} must be 1–{limit} characters")
        return value

    def execute(self, action):
        """Execute one validated, named Workspace operation. No arbitrary URL, HTTP method or API body is accepted."""
        kind = action.get("action")
        if kind == "workspace_sheets_write":
            spreadsheet = self._id(action.get("spreadsheet_id"), "spreadsheet id")
            cell = str(action.get("cell") or "").strip()
            if not re.fullmatch(r"(?:(?:'[^']{1,80}'|[\w .-]{1,80})!)?[A-Z]{1,3}[1-9][0-9]{0,6}", cell):
                raise WorkspaceError("use one A1 cell, for example Sheet1!B4")
            value = self._text(action.get("value"), "cell value", 2000)
            quoted = urllib.parse.quote(cell, safe="")
            base = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet}/values/{quoted}"
            current = self._api("GET", base, SCOPES["sheets"])
            values = current.get("values") if isinstance(current, dict) else None
            occupied = bool(values and any(str(v).strip() for row in values if isinstance(row, list) for v in row))
            if occupied and action.get("overwrite") is not True:
                raise WorkspaceError("the cell is not empty; approve an overwrite before writing")
            out = self._api("PUT", base + "?valueInputOption=USER_ENTERED", SCOPES["sheets"],
                            {"range": cell, "majorDimension": "ROWS", "values": [[value]]})
            return {"ok": True, "service": "sheets", "updated": out.get("updatedRange") or cell,
                    "_undo": {"kind": "sheets", "spreadsheet_id": spreadsheet, "cell": cell,
                              "values": values or []}, "_undo_label": f"Restore {cell}"}
        if kind == "workspace_docs_append":
            document = self._id(action.get("document_id"), "document id")
            value = self._text(action.get("text"), "document text", 10_000)
            url = f"https://docs.googleapis.com/v1/documents/{document}"
            doc = self._api("GET", url, SCOPES["docs"])
            content = ((doc.get("body") or {}).get("content") or []) if isinstance(doc, dict) else []
            indexes = [x.get("endIndex") for x in content if isinstance(x, dict) and isinstance(x.get("endIndex"), int)]
            if not indexes: raise WorkspaceError("Google Docs did not return a writable document body")
            start = max(1, max(indexes) - 1)
            self._api("POST", url + ":batchUpdate", SCOPES["docs"],
                      {"requests": [{"insertText": {"location": {"index": start}, "text": value}}]})
            units = len(value.encode("utf-16-le")) // 2
            return {"ok": True, "service": "docs", "document_id": document,
                    "_undo": {"kind": "docs", "document_id": document, "start": start, "end": start + units},
                    "_undo_label": "Remove appended text"}
        if kind == "workspace_drive_create":
            name = self._text(action.get("name"), "file name", 200)
            value = self._text(action.get("text"), "document text", 20_000)
            made = self._api("POST", "https://www.googleapis.com/drive/v3/files?fields=id,name,mimeType,webViewLink",
                             SCOPES["drive"], {"name": name, "mimeType": "application/vnd.google-apps.document"})
            document = self._id(made.get("id") if isinstance(made, dict) else None, "created document id")
            self._api("POST", f"https://docs.googleapis.com/v1/documents/{document}:batchUpdate", SCOPES["docs"],
                      {"requests": [{"insertText": {"location": {"index": 1}, "text": value}}]})
            return {"ok": True, "service": "drive", "file_id": document, "name": name,
                    "url": made.get("webViewLink") if isinstance(made, dict) else None,
                    "_undo": {"kind": "drive", "file_id": document}, "_undo_label": f"Move “{name}” to Trash"}
        if kind == "workspace_gmail_draft":
            raw_to = str(action.get("to") or "")
            if "\r" in raw_to or "\n" in raw_to:
                raise WorkspaceError("email recipients cannot contain line breaks")
            recipients = email.utils.getaddresses([raw_to])
            if not recipients or len(recipients) > 10 or any(not addr or "@" not in addr for _, addr in recipients):
                raise WorkspaceError("enter valid email recipients")
            subject = self._text(action.get("subject"), "subject", 300)
            if "\r" in subject or "\n" in subject:
                raise WorkspaceError("email subject cannot contain line breaks")
            value = self._text(action.get("body"), "message body", 20_000)
            try:
                msg = email.message.EmailMessage(); msg["To"] = ", ".join(addr for _, addr in recipients)
                msg["Subject"] = subject; msg.set_content(value)
            except (TypeError, ValueError):
                raise WorkspaceError("invalid email headers") from None
            raw = base64.urlsafe_b64encode(msg.as_bytes()).decode().rstrip("=")
            made = self._api("POST", "https://gmail.googleapis.com/gmail/v1/users/me/drafts",
                             SCOPES["gmail_compose"], {"message": {"raw": raw}})
            draft = self._id(made.get("id") if isinstance(made, dict) else None, "draft id")
            return {"ok": True, "service": "gmail", "draft_id": draft,
                    "_undo": {"kind": "gmail", "draft_id": draft}, "_undo_label": "Delete Gmail draft"}
        if kind == "workspace_calendar_create":
            summary = self._text(action.get("summary"), "event title", 300)
            start, end = str(action.get("start") or ""), str(action.get("end") or "")
            try:
                a = datetime.datetime.fromisoformat(start.replace("Z", "+00:00"))
                b = datetime.datetime.fromisoformat(end.replace("Z", "+00:00"))
            except ValueError: raise WorkspaceError("event times must be ISO 8601 with a time zone") from None
            if a.tzinfo is None or b.tzinfo is None or b <= a or b - a > datetime.timedelta(days=7):
                raise WorkspaceError("event end must follow start, include a time zone and be within seven days")
            zone = str(action.get("time_zone") or "")[:80]
            event = {"summary": summary, "start": {"dateTime": start}, "end": {"dateTime": end}}
            if zone: event["start"]["timeZone"] = event["end"]["timeZone"] = zone
            made = self._api("POST", "https://www.googleapis.com/calendar/v3/calendars/primary/events",
                             SCOPES["calendar"], event)
            event_id = self._id(made.get("id") if isinstance(made, dict) else None, "event id")
            return {"ok": True, "service": "calendar", "event_id": event_id,
                    "url": made.get("htmlLink") if isinstance(made, dict) else None,
                    "_undo": {"kind": "calendar", "event_id": event_id}, "_undo_label": "Delete Calendar event"}
        raise WorkspaceError("unsupported Workspace operation")

    def undo(self, record):
        """Reverse one service-created operation. Records are kept only in service memory behind opaque tokens."""
        kind = record.get("kind") if isinstance(record, dict) else None
        if kind == "sheets":
            spreadsheet = self._id(record.get("spreadsheet_id"), "spreadsheet id")
            cell = str(record.get("cell") or "")
            url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet}/values/{urllib.parse.quote(cell, safe='')}"
            values = record.get("values")
            if values:
                self._api("PUT", url + "?valueInputOption=RAW", SCOPES["sheets"],
                          {"range": cell, "majorDimension": "ROWS", "values": values})
            else:
                self._api("POST", url + ":clear", SCOPES["sheets"], {})
        elif kind == "docs":
            document = self._id(record.get("document_id"), "document id")
            start, end = int(record.get("start") or 0), int(record.get("end") or 0)
            if start < 1 or end <= start: raise WorkspaceError("invalid Docs undo range")
            self._api("POST", f"https://docs.googleapis.com/v1/documents/{document}:batchUpdate", SCOPES["docs"],
                      {"requests": [{"deleteContentRange": {"range": {"startIndex": start, "endIndex": end}}}]})
        elif kind == "drive":
            file_id = self._id(record.get("file_id"), "file id")
            self._api("PATCH", f"https://www.googleapis.com/drive/v3/files/{file_id}?fields=id,trashed",
                      SCOPES["drive"], {"trashed": True})
        elif kind == "gmail":
            draft = self._id(record.get("draft_id"), "draft id")
            self._api("DELETE", f"https://gmail.googleapis.com/gmail/v1/users/me/drafts/{draft}",
                      SCOPES["gmail_compose"])
        elif kind == "calendar":
            event = self._id(record.get("event_id"), "event id")
            self._api("DELETE", f"https://www.googleapis.com/calendar/v3/calendars/primary/events/{event}",
                      SCOPES["calendar"])
        else:
            raise WorkspaceError("this operation cannot be undone")
        return {"ok": True, "undone": kind}


def html_escape(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
