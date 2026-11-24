"""The code editor's chat history file (harness/chats.py): its format, as the Swift decoder needs it.

    .venv/bin/python -m unittest tests.test_chats -v
"""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import chats  # noqa: E402

ITEM_KEYS = {"id", "kind", "text", "reasoning", "attached", "steps", "counts"}
COUNT_KEYS = {"searches", "explores", "reads", "edits", "commands", "questions", "webs", "images"}


class TestFormat(unittest.TestCase):
    def test_file_name_matches_the_editors_key(self):
        # the editor's file for this very project was written by Swift: <name>-<djb2 hex>.json
        self.assertEqual(chats.project_key("/Users/denis/huggingface"), "huggingface-cf406f8d375acc0f.json")
        self.assertEqual(chats.project_key("/"), f"-{(5381 * 33 + 47):x}.json")

    def test_a_session_has_every_field_the_swift_decoder_requires(self):
        s = chats.new_session()
        chats.add_turn(s, "user", "hello", attached=["/a.png"])
        chats.add_turn(s, "assistant", "hi", "thought")
        self.assertEqual(set(s), {"id", "title", "updated", "items"})
        for item in s["items"]:
            self.assertEqual(set(item), ITEM_KEYS)
            self.assertEqual(set(item["counts"]), COUNT_KEYS)
        self.assertEqual(s["id"], s["id"].upper())
        self.assertGreater(s["updated"], 7e8)               # seconds since 2001, not 1970
        self.assertLess(s["updated"], 4e9)
        self.assertEqual(s["items"][0]["attached"], ["/a.png"])

    def test_title_comes_from_the_first_question_and_is_cut(self):
        s = chats.new_session()
        chats.add_turn(s, "user", "What is the weather like\n in Lisbon in May? And what should I pack for it?")
        self.assertEqual(s["title"], "What is the weather like in Lisbon in May? And …")
        chats.add_turn(s, "user", "another")
        self.assertTrue(s["title"].startswith("What is"))

    def test_messages_skip_cards_and_keep_reasoning(self):
        s = {"id": "A", "title": "t", "updated": 1, "items": [
            {"kind": "user", "text": "q"}, {"kind": "activity", "text": ""},
            {"kind": "assistant", "text": "a", "reasoning": "r"}, {"kind": "error", "text": "boom"},
            {"kind": "image", "text": "x"}, {"kind": "assistant", "text": ""}]}
        self.assertEqual(chats.messages(s), [{"role": "user", "content": "q"},
                                             {"role": "assistant", "content": "a", "reasoning_content": "r"}])


class TestStore(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.root = "/work/proj"

    def make(self, n):
        out = []
        for i in range(n):
            s = chats.new_session()
            chats.add_turn(s, "user", f"q{i}")
            s["updated"] = 1000 + i
            out.append(s)
        return out

    def test_roundtrip_newest_first_and_empty_chats_are_not_stored(self):
        ss = self.make(3) + [chats.new_session()]
        chats.save(self.root, ss, self.dir)
        got = chats.load(self.root, self.dir)
        self.assertEqual([s["title"] for s in got], ["q2", "q1", "q0"])
        with open(chats.chats_file(self.root, self.dir), encoding="utf-8") as f:
            self.assertIsInstance(json.load(f), list)
        self.assertEqual([n for n in os.listdir(self.dir) if n.endswith(".tmp")], [])

    def test_keeps_forty(self):
        chats.save(self.root, self.make(45), self.dir)
        got = chats.load(self.root, self.dir)
        self.assertEqual(len(got), 40)
        self.assertEqual(got[-1]["title"], "q5")

    def test_damaged_or_foreign_files_are_an_empty_history(self):
        path = chats.chats_file(self.root, self.dir)
        for body in ("", "{not json", '{"a": 1}', '"x"', '[1, "a", null]'):
            with open(path, "w") as f:
                f.write(body)
            self.assertEqual(chats.load(self.root, self.dir), [], body)
        with open(path, "w") as f:
            json.dump([{"id": "X", "title": 5, "items": [3, {"kind": "user", "text": "q"}]}], f)
        s = chats.load(self.root, self.dir)[0]
        self.assertEqual((s["title"], s["updated"], len(s["items"])), ("New chat", 0, 1))

    def test_editors_extra_items_survive_a_cli_turn(self):
        s = chats.new_session()
        s["items"] = [{"id": "I1", "kind": "approval", "text": "", "reasoning": "", "attached": [], "steps": [],
                       "counts": {}, "approval": {"id": "a", "kind": "edit", "path": "x.py"}}]
        chats.add_turn(s, "user", "q")
        chats.save(self.root, [s], self.dir)
        got = chats.load(self.root, self.dir)[0]
        self.assertEqual(got["items"][0]["approval"]["path"], "x.py")
        self.assertEqual(len(got["items"]), 2)

    def test_find_rename_delete(self):
        ss = chats.load(self.root, self.dir)
        ss[:] = sorted(self.make(3), key=lambda s: -s["updated"])
        self.assertEqual(chats.find(ss, "1")["title"], "q2")
        self.assertIsNone(chats.find(ss, "0"))
        self.assertIsNone(chats.find(ss, "4"))
        self.assertIsNone(chats.find(ss, ""))
        self.assertEqual(chats.find(ss, ss[1]["id"][:8])["title"], "q1")
        self.assertIsNone(chats.find(ss, "ZZZZ"))
        self.assertFalse(chats.rename(ss[0], "   "))
        self.assertTrue(chats.rename(ss[0], "  new   name "))
        self.assertEqual(ss[0]["title"], "new name")
        chats.delete(ss, ss[0])
        self.assertEqual([s["title"] for s in ss], ["q1", "q0"])
        self.assertEqual(len(chats.listing(ss, ss[0]["id"])), 2)
        self.assertTrue(chats.listing(ss, ss[0]["id"])[0].startswith("▸"))


if __name__ == "__main__":
    unittest.main()
