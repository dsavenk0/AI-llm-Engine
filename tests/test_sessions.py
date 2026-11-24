"""The terminal chat's saved conversations (harness/sessions.py) and the chat around them (agent.py): many chats,
each written as it goes, continued, listed, renamed and deleted; a project attached on demand; the command line.

    .venv/bin/python -m unittest tests.test_sessions -v
"""
import contextlib
import io
import json
import os
import re
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("LLM_SESSIONS_DIR", tempfile.mkdtemp(prefix="llm-test-chats-"))  # never the real chats
import agent  # noqa: E402
import llm  # noqa: E402
from harness import sessions  # noqa: E402


class TestStore(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def chat(self, project=None):
        return sessions.new(project, self.dir)

    def test_nothing_is_written_for_an_empty_chat(self):
        c = self.chat()
        c.sync([])
        self.assertEqual(os.listdir(self.dir), [])

    def test_messages_are_appended_as_they_come_and_read_back(self):
        c = self.chat("/work/p")
        c.sync([{"role": "user", "content": "How do I list files?"}])
        c.sync([{"role": "user", "content": "How do I list files?"}, {"role": "assistant", "content": "ls"}])
        with open(c.path) as f:
            lines = [json.loads(l) for l in f]
        self.assertEqual(len(lines), 3)                                # header + one line per message, no rewrite
        back = sessions.read(c.path)
        self.assertEqual((back.title, back.project, back.questions), ("How do I list files?", "/work/p", 1))
        self.assertEqual([m["content"] for m in back.messages], ["How do I list files?", "ls"])

    def test_a_changed_conversation_is_one_reset_line(self):
        c = self.chat()
        c.sync([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])
        c.sync([{"role": "user", "content": "summary of a and b"}])         # compaction replaced it
        self.assertEqual([m["content"] for m in sessions.read(c.path).messages], ["summary of a and b"])

    def test_the_system_prompt_is_not_stored(self):
        c = self.chat()
        c.sync([{"role": "system", "content": "secret"}, {"role": "user", "content": "q"}])
        with open(c.path) as f:
            self.assertNotIn("secret", f.read())

    def test_a_chat_cut_mid_turn_is_repaired_and_damaged_lines_skipped(self):
        c = self.chat()
        c.sync([{"role": "user", "content": "q"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "x1", "function": {"name": "run"}},
                                                                   {"id": "x2", "function": {"name": "run"}}]},
                {"role": "tool", "tool_call_id": "x1", "content": "ok"}])
        with open(c.path, "a") as f:
            f.write('{"m": {"role": "tool", "tool_ca')                    # the crash cut this line
        back = sessions.read(c.path)
        self.assertEqual([m.get("tool_call_id") for m in back.messages if m["role"] == "tool"], ["x1", "x2"])
        self.assertIn("cancelled", back.messages[-1]["content"])

    def test_a_repaired_chat_is_written_back_whole_not_duplicated(self):
        c = self.chat()
        c.sync([{"role": "user", "content": "q"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "x1", "function": {"name": "run"}},
                                                                   {"id": "x2", "function": {"name": "run"}}]},
                {"role": "tool", "tool_call_id": "x1", "content": "ok"},
                {"role": "user", "content": "next"}, {"role": "assistant", "content": "a2"}])
        back = sessions.read(c.path)                       # x2 never answered: repaired in the middle
        self.assertTrue(back.stale)
        back.sync(back.messages + [{"role": "user", "content": "more"}])
        again = sessions.read(c.path)
        self.assertEqual([m.get("content") for m in again.messages],
                         ["q", "", "ok", "cancelled (the chat was closed)", "next", "a2", "more"])
        self.assertFalse(again.stale)

    def test_list_find_rename_delete(self):
        a, b = self.chat(), self.chat("/work/x")
        a.sync([{"role": "user", "content": "first"}])
        time.sleep(0.01)
        b.sync([{"role": "user", "content": "second"}])
        chats = sessions.all_chats(self.dir)
        self.assertEqual([c.title for c in chats], ["second", "first"])      # newest first
        self.assertEqual(sessions.find(chats, "2").id, a.id)
        self.assertEqual(sessions.find(chats, a.id[:-2]).id, a.id)
        self.assertIsNone(sessions.find(chats, "9"))
        self.assertIsNone(sessions.find(chats, ""))
        lines = sessions.listing(sessions.all_chats(self.dir), current=b.id)
        self.assertTrue(lines[0].startswith("▸") and "· x" in lines[0])
        a.rename("Renamed")
        self.assertEqual(sessions.read(a.path).title, "Renamed")
        sessions.delete(sessions.find(sessions.all_chats(self.dir), "1"))       # the renamed one is newest now
        self.assertEqual([c.title for c in sessions.all_chats(self.dir)], ["second"])

    def test_latest_prefers_the_project(self):
        a, b = self.chat("/work/p"), self.chat()
        a.sync([{"role": "user", "content": "in p"}])
        time.sleep(0.01)
        b.sync([{"role": "user", "content": "elsewhere"}])
        self.assertEqual(sessions.latest(self.dir, project="/work/p").title, "in p")
        self.assertEqual(sessions.latest(self.dir, project="/work/none").title, "elsewhere")
        self.assertEqual(sessions.latest(self.dir).title, "elsewhere")

    def test_prune_drops_old_and_extra_chats(self):
        made = []
        for i in range(5):
            c = self.chat()
            c.sync([{"role": "user", "content": f"q{i}"}])
            os.utime(c.path, (time.time() - (40 - i) * 86400,) * 2 if i < 2 else None)
            made.append(c)
        with mock.patch.object(sessions, "KEEP_CHATS", 2):
            self.assertEqual(sessions.prune(self.dir), 3)               # two too old + one beyond the limit
        self.assertEqual(sorted(c.title for c in sessions.all_chats(self.dir)), ["q3", "q4"])


class TestParse(unittest.TestCase):
    def test_flags(self):
        p = agent.parse
        self.assertEqual(p([])["prompt"], None)
        self.assertEqual(p(["what", "is", "this"])["prompt"], "what is this")
        self.assertTrue(p(["-c"])["continue"])
        self.assertEqual(p(["-r"])["resume"], "")
        self.assertEqual(p(["-r", "3"])["resume"], "3")
        self.assertEqual(p(["--resume", "20261124-101010-abc123"])["resume"], "20261124-101010-abc123")
        o = p(["-r", "my chat", "and a question"])                        # like `claude -r <session> "query"`
        self.assertEqual((o["resume"], o["prompt"]), ("my chat", "and a question"))
        o = p(["-c", "--fork-session", "-n", "Plan B", "--no-session-persistence", "--output-format=json",
               "--max-turns", "3", "--append-system-prompt", "Be brief."])
        self.assertEqual((o["fork"], o["name"], o["save"], o["format"], o["max_turns"], o["append"]),
                         (True, "Plan B", False, "json", 3, "Be brief."))
        o = p(["-p", "sum it", "--project", "src", "--auto", "-m", "qwen", "--no-think"])
        self.assertEqual((o["print"], o["project"], o["auto"], o["model"], o["think"]),
                         ("sum it", "src", True, "qwen", False))
        for bad in (["-p"], ["--project"], ["--frobnicate"], ["--output-format", "xml"], ["--max-turns", "0"]):
            with self.subTest(bad), self.assertRaises(ValueError):
                p(bad)


def script(*replies):
    """A fake streamed reply per request: plain text, or a tool call given as (name, args)."""
    out = []
    for r in replies:
        if isinstance(r, tuple):
            ev = {"choices": [{"delta": {"tool_calls": [{"id": f"c{len(out)}", "type": "function",
                                                         "function": {"name": r[0], "arguments": json.dumps(r[1])}}]}}]}
        else:
            ev = {"choices": [{"delta": {"content": r}}]}
        out.append([b"data: " + json.dumps(ev).encode(), b"data: [DONE]"])
    return out


class Stream:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(self.lines)


class ChatCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.here = os.path.realpath(tempfile.mkdtemp())
        with open(os.path.join(self.here, "notes.txt"), "w") as f:
            f.write("the code word is heron\n")
        self.sent = []
        self.env = mock.patch.dict(os.environ, {"LLM_SESSIONS_DIR": self.dir, "LLM_CWD": self.here})
        self.env.start()
        self.addCleanup(self.env.stop)

    def run_chat(self, lines, replies=(), chat=None, root=None, answers=()):
        """Drive Agent.repl with typed lines; `answers` are the y/n replies to its questions."""
        typed, replies, yes = iter(lines), list(script(*replies)), iter(answers)

        def fake_input(prompt=""):
            try:
                return next(typed)
            except StopIteration:
                raise EOFError

        def fake_urlopen(req, timeout=None):
            body = json.loads(req.data)
            self.sent.append(body)
            return Stream(replies.pop(0) if replies else script("(end)")[0])

        cfg = dict(llm.load_cfg(), port=1)
        buf = io.StringIO()
        with mock.patch("builtins.input", fake_input), mock.patch("urllib.request.urlopen", fake_urlopen), \
                mock.patch.object(llm, "health", return_value=None), \
                mock.patch.object(llm, "ask_yn", side_effect=lambda *a, **k: next(yes)), \
                contextlib.redirect_stdout(buf):
            a = agent.Agent(cfg, root, chat)
            a.repl()
            a.persist()
        self.agent = a
        return buf.getvalue()


class TestChat(ChatCase):
    def test_a_chat_is_saved_then_continued_with_its_context(self):
        self.run_chat(["first question", "second one"], ["answer 1", "answer 2"])
        saved = sessions.all_chats()
        self.assertEqual((len(saved), saved[0].title, saved[0].questions), (1, "first question", 2))
        self.sent.clear()
        self.run_chat(["third"], ["answer 3"], chat=saved[0])
        roles = [(m["role"], m["content"]) for m in self.sent[0]["messages"][1:]]
        self.assertEqual(roles, [("user", "first question"), ("assistant", "answer 1"), ("user", "second one"),
                                 ("assistant", "answer 2"), ("user", "third")])
        self.assertEqual(sessions.all_chats()[0].questions, 3)

    def test_chats_open_rename_delete_from_inside_a_chat(self):
        self.run_chat(["old topic"], ["a"])
        self.sent.clear()
        out = self.run_chat(["new topic", "/chats", "/open 2", "back to it", "/rename Kept", "/delete 1"],
                            ["b", "c"], answers=[False])
        self.assertIn("old topic", out)
        self.assertEqual([m["content"] for m in self.sent[1]["messages"][1:]], ["old topic", "a", "back to it"])
        self.assertEqual(sorted(c.title for c in sessions.all_chats()), ["Kept", "new topic"])   # delete said no
        out = self.run_chat(["/delete 2"], answers=[True])
        self.assertEqual(len(sessions.all_chats()), 1)

    def test_new_keeps_the_old_chat_and_wrong_numbers_change_nothing(self):
        out = self.run_chat(["one", "/new", "two", "/open 9", "/delete x"], ["a", "b"])
        self.assertEqual([m["content"] for m in self.sent[1]["messages"][1:]], ["two"])
        self.assertEqual(sorted(c.title for c in sessions.all_chats()), ["one", "two"])
        self.assertEqual(out.count(llm.T("No such chat. /chats lists them.")), 2)

    def test_without_a_project_there_are_no_file_tools_until_one_is_attached(self):
        self.run_chat(["hi", "/project", "and now?"], ["hello", "ready"])
        first = {t["function"]["name"] for t in self.sent[0]["tools"]}
        self.assertIn("attach_project", first)
        self.assertFalse({"read_file", "run", "edit_file"} & first)
        second = {t["function"]["name"] for t in self.sent[1]["tools"]}
        self.assertTrue({"read_file", "run", "edit_file"} <= second)
        self.assertEqual(sessions.all_chats()[0].project, self.here)    # the chat remembers its project

    def test_the_model_asks_for_the_project_and_the_user_decides(self):
        self.run_chat(["what is the code word in my notes?"],
                      [("attach_project", {}), ("read_file", {"path": "notes.txt"}), "heron"], answers=[True])
        self.assertEqual(self.agent.root, self.here)
        tool_results = [m["content"] for m in self.sent[-1]["messages"] if m["role"] == "tool"]
        self.assertIn("attached", tool_results[0])
        self.assertIn("heron", tool_results[1])

        self.sent.clear()
        self.run_chat(["and again?"], [("attach_project", {}), "fine"], answers=[False])
        self.assertIsNone(self.agent.root)
        self.assertIn("declined", [m for m in self.sent[-1]["messages"] if m["role"] == "tool"][0]["content"])

    def test_retry_does_not_attach_files_twice(self):
        self.run_chat(["what is in @notes.txt?", "/retry"], ["first", "second"])
        sent = self.sent[1]["messages"][-1]["content"]
        self.assertEqual(sent.count('<file path="notes.txt">'), 1)

    def test_a_chat_whose_project_is_gone_keeps_it_and_runs_without_files(self):
        gone = os.path.join(self.here, "was-here")
        os.makedirs(gone)
        self.run_chat(["/project was-here", "hi"], ["hello"])
        os.rmdir(gone)
        chat = sessions.all_chats()[0]
        out = self.run_chat(["/open 1", "again"], ["ok"], root=None)
        self.assertIsNone(self.agent.root)
        self.assertIn("was-here", out)
        self.assertEqual(sessions.all_chats()[0].project, gone)          # remembered for when it is back
        self.assertEqual(chat.project, gone)

    def test_project_commands_need_a_project(self):
        out = self.run_chat(["/memory", "/project off"])
        self.assertIn(llm.T("This needs a project: /project attaches {p}").format(p=self.here), out)

    def test_retry_asks_the_last_question_again_without_the_old_answer(self):
        self.run_chat(["question", "/retry"], ["bad answer", "better answer"])
        self.assertEqual([m["content"] for m in self.sent[1]["messages"][1:]], ["question"])
        self.assertEqual([m["content"] for m in sessions.all_chats()[0].messages], ["question", "better answer"])

    def test_several_lines_at_once(self):
        self.run_chat(['"""', "line one", "line two", '"""'], ["ok"])
        self.assertEqual(self.sent[0]["messages"][-1]["content"], "line one\nline two")

    def test_a_single_ctrl_c_at_the_prompt_does_not_leave(self):
        typed = iter([KeyboardInterrupt, "still here", KeyboardInterrupt, KeyboardInterrupt, "not reached"])

        def fake_input(prompt=""):
            v = next(typed)
            if v is KeyboardInterrupt:
                raise KeyboardInterrupt
            return v
        cfg = dict(llm.load_cfg(), port=1)
        with mock.patch("builtins.input", fake_input), \
                mock.patch("urllib.request.urlopen", lambda req, timeout=None: Stream(script("ok")[0])), \
                mock.patch.object(llm, "health", return_value=None), contextlib.redirect_stdout(io.StringIO()):
            a = agent.Agent(cfg)
            a.repl()
        self.assertEqual([m["content"] for m in a.messages[1:]], ["still here", "ok"])
        self.assertEqual(next(typed), "not reached")


class TestCommandLine(ChatCase):
    def main(self, args, replies=(), stdin_tty=True, stdin_text=""):
        replies = list(script(*replies))

        def fake_urlopen(req, timeout=None):
            self.sent.append(json.loads(req.data))
            return Stream(replies.pop(0) if replies else script("(end)")[0])
        out, err = io.StringIO(), io.StringIO()
        fake_stdin = io.StringIO(stdin_text)
        fake_stdin.isatty = lambda: stdin_tty
        with mock.patch("urllib.request.urlopen", fake_urlopen), \
                mock.patch.object(llm, "set_lang"), \
                mock.patch.object(llm, "health", return_value={"context": 8000}), \
                mock.patch("builtins.input", side_effect=EOFError), mock.patch.object(sys, "stdin", fake_stdin), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            ok = agent.main(args)
        return ok, out.getvalue(), err.getvalue()

    def test_print_mode_writes_only_the_answer(self):
        ok, out, err = self.main(["-p", "say hi"], ["hi there"], stdin_tty=False, stdin_text="")
        self.assertTrue(ok)
        self.assertEqual(out, "hi there\n")
        self.assertEqual(sessions.all_chats()[0].title, "say hi")          # saved like any chat

    def test_print_mode_reads_a_pipe_and_refuses_what_needs_a_yes(self):
        ok, out, _ = self.main(["-p", "fix it", "--project", self.here], [("write_file", {"path": "x", "content": "y"}),
                                                                           "could not write"],
                               stdin_tty=False, stdin_text="Traceback: boom")
        self.assertIn("Traceback: boom", self.sent[0]["messages"][1]["content"])
        self.assertFalse(os.path.exists(os.path.join(self.here, "x")))
        self.assertEqual(out, "could not write\n")

    def test_continue_and_resume(self):
        self.main(["first"], ["a"])
        self.sent.clear()
        self.main(["-c"])                                             # no new question: opens and waits
        self.main(["-c", "more"], ["b"])
        self.assertEqual([m["content"] for m in self.sent[0]["messages"][1:]], ["first", "a", "more"])
        ok, out, _ = self.main(["-r", "9"])
        self.assertFalse(ok)
        self.assertIn(llm.T("No such chat. `llm chats` lists them."), out)

    def test_chats_command_lists_and_deletes(self):
        self.main(["keep me"], ["a"])
        self.main(["drop me"], ["b"])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            agent.chats_cmd([])
            agent.chats_cmd(["rm", "1", "--yes"])
        self.assertIn("drop me", out.getvalue())
        self.assertEqual([c.title for c in sessions.all_chats()], ["keep me"])

    def test_fork_name_and_resume_by_name(self):
        self.main(["-n", "Release plan", "first"], ["a"])
        self.assertEqual(sessions.all_chats()[0].title, "Release plan")
        self.sent.clear()
        self.main(["-r", "release", "--fork-session", "more"], ["b"])   # a unique beginning of the name
        self.assertEqual([m["content"] for m in self.sent[0]["messages"][1:]], ["first", "a", "more"])
        chats = sessions.all_chats()
        self.assertEqual(sorted((c.title, c.questions) for c in chats),
                         [("Release plan", 1), ("Release plan (fork)", 2)])     # the original is untouched

    def test_json_output_no_persistence_and_max_turns(self):
        ok, out, _ = self.main(["-p", "hi", "--output-format", "json", "--no-session-persistence"], ["hello"],
                               stdin_tty=False)
        res = json.loads(out)
        self.assertEqual((ok, res["result"], res["is_error"], res["type"]), (True, "hello", False, "result"))
        self.assertEqual(sessions.all_chats(), [])
        ok, out, _ = self.main(["-p", "go", "--project", self.here, "--max-turns", "2", "--output-format", "json"],
                               [("list_dir", {"path": "."}), ("read_file", {"path": "notes.txt"}), "never"],
                               stdin_tty=False)
        self.assertEqual(len(self.sent), 3)                           # 1 + the two capped replies
        self.assertTrue(json.loads(out)["is_error"])


class TestChatExtras(ChatCase):
    def test_at_file_adds_the_file_text(self):
        self.run_chat(["what is in @notes.txt?"], ["heron"])
        content = self.sent[0]["messages"][-1]["content"]
        self.assertIn('<file path="notes.txt">', content)
        self.assertIn("heron", content)
        self.run_chat(["mail me @someone"], ["ok"])                   # not a file: left as typed
        self.assertEqual(self.sent[-1]["messages"][-1]["content"], "mail me @someone")

    def _drop_tree(self):
        """A folder and files with the names a drag-and-drop has to escape: spaces, a quote, Cyrillic."""
        folder = os.path.join(self.here, "My Project")
        os.makedirs(folder)
        text = os.path.join(self.here, "кот's notes.txt")
        with open(text, "w") as f:
            f.write("the cat is called Ash\n")
        picture = os.path.join(self.here, "shot 1.png")
        with open(picture, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")
        binary = os.path.join(self.here, "data.bin")
        with open(binary, "wb") as f:
            f.write(b"\xff\xfe\x00\x81" * 10)
        esc = lambda p: re.sub(r"([^\w/.\-+,:@%=~])", r"\\\1", p)       # as Terminal.app and the editor type it
        return folder, text, picture, binary, esc

    def test_dragged_paths_are_read_as_a_terminal_types_them(self):
        folder, text, picture, _, esc = self._drop_tree()
        self.assertEqual(agent.dropped(esc(folder) + " "), ([folder], ""))
        self.assertEqual(agent.dropped(f'"{text}" {esc(picture)} what are these?'),
                         ([text, picture], "what are these?"))
        self.assertEqual(agent.dropped(f"'{folder}'"), ([folder], ""))
        for line in ("what is /tmp for?", "notes.txt please", "/nowhere/at/all x", "/help", "it's fine"):
            self.assertEqual(agent.dropped(line), ([], line), line)

    def test_a_dragged_folder_opens_as_the_project_when_the_user_says_yes(self):
        folder, *_, esc = self._drop_tree()
        self.run_chat([esc(folder) + " "], answers=[False])
        self.assertIsNone(self.agent.root)
        self.assertEqual(self.sent, [])                                 # nothing went to the model
        self.run_chat([esc(folder) + " ", "hi"], ["hello"], answers=[True])
        self.assertEqual(self.agent.root, folder)
        self.assertEqual(sessions.latest().project, folder)

    def test_dragged_files_go_with_the_text_or_with_the_next_message(self):
        _, text, picture, binary, esc = self._drop_tree()
        self.run_chat([f"{esc(text)} what is the cat called?"], ["Ash"])
        content = self.sent[-1]["messages"][-1]["content"]
        self.assertTrue(content.startswith("what is the cat called?"))
        self.assertIn(f'<file path="{text}">', content)
        self.assertIn("Ash", content)

        out = self.run_chat([f"{esc(picture)} {esc(binary)} ", "and this?"], ["a screenshot"])
        self.assertEqual(len(self.sent), 2)                             # the drop alone sent nothing
        self.assertIn(llm.T("Not attached: {n} is not a text file under 250 KB.").format(n="data.bin"), out)
        content = self.sent[-1]["messages"][-1]["content"]
        self.assertEqual(content[0], {"type": "image_url", "image_url": {"url": picture}})
        self.assertEqual(content[-1], {"type": "text", "text": "and this?"})

    def test_fork_and_status_inside_a_chat(self):
        out = self.run_chat(["topic", "/fork", "/status", "more"], ["a", "b"])
        chats = sessions.all_chats()
        self.assertEqual(sorted((c.title, c.questions) for c in chats), [("topic", 1), ("topic (fork)", 2)])
        self.assertIn("topic (fork)", out)


if __name__ == "__main__":
    unittest.main()
