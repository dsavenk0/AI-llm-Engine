"""Saved skills (harness/skills.py): the Agent Skills format, where they are found, what the model sees, the chat
commands that manage them, and the editor assistant's tools for them.

    .venv/bin/python -m unittest tests.test_skills -v
"""
import os
import tempfile
import threading
import unittest
from unittest import mock

from harness import skills
os.environ.setdefault("LLM_SESSIONS_DIR", __import__("tempfile").mkdtemp(prefix="llm-test-chats-"))  # never the real chats


def write_skill(base, folder, text, extra=None):
    d = os.path.join(base, folder)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "SKILL.md"), "w") as f:
        f.write(text)
    for rel, body in (extra or {}).items():
        os.makedirs(os.path.dirname(os.path.join(d, rel)), exist_ok=True)
        with open(os.path.join(d, rel), "w") as f:
            f.write(body)
    return d


class Env(unittest.TestCase):
    """A project and a home of its own, so the user's real ~/.llm, ~/.codex and ~/.claude are never read."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="skills-project-")
        self.home = tempfile.mkdtemp(prefix="skills-home-")
        p = mock.patch.dict(os.environ, {"HOME": self.home, "LLM_SKILLS_DIRS": os.pathsep.join(
            [os.path.join(self.home, ".codex/skills"), os.path.join(self.home, ".claude/skills")])})
        p.start()
        self.addCleanup(p.stop)

    def project(self, folder, text, where=".llm/skills", **kw):
        return write_skill(os.path.join(self.root, where), folder, text, **kw)

    def personal(self, folder, text, where=".llm/skills", **kw):
        return write_skill(os.path.join(self.home, where), folder, text, **kw)


class TestFormat(unittest.TestCase):
    def test_front_matter_with_quotes_blocks_and_nested_keys(self):
        meta, body = skills.parse('---\nname: pdf-tools\ndescription: "Fill PDF forms: when asked"\n'
                                  'notes: >\n  folded\n  text\nmetadata:\n  short-description: Short\n---\n\n# Body\n')
        self.assertEqual(meta["name"], "pdf-tools")
        self.assertEqual(meta["description"], "Fill PDF forms: when asked")
        self.assertEqual(meta["notes"], "folded text")
        self.assertEqual(meta["metadata.short-description"], "Short")
        self.assertEqual(body, "# Body\n")

    def test_no_front_matter_is_all_body(self):
        self.assertEqual(skills.parse("just text"), ({}, "just text"))

    def test_what_is_saved_reads_back_the_same(self):
        text = skills.render("deploy", 'Ship it "now"', "1. build\n2. ship")
        meta, body = skills.parse(text)
        self.assertEqual((meta["name"], meta["description"], body.strip()), ("deploy", "Ship it 'now'", "1. build\n2. ship"))


class TestDiscovery(Env):
    def test_project_wins_over_personal_and_other_tools_are_read_only(self):
        self.project("review", "---\nname: review\ndescription: project review\n---\nP")
        self.personal("review", "---\nname: review\ndescription: personal review\n---\nQ")
        self.personal("lint", "---\nname: lint\ndescription: from codex\n---\nL", where=".codex/skills")
        found = {s.name: s for s in skills.discover(self.root)}
        self.assertEqual(found["review"].description, "project review")
        self.assertEqual(found["review"].scope, "project")
        self.assertFalse(found["lint"].writable)
        self.assertEqual(skills.find("$lint", self.root).description, "from codex")

    def test_other_tools_personal_folders_need_to_be_named(self):
        self.personal("lint", "---\nname: lint\ndescription: d\n---\n", where=".claude/skills")
        self.assertIsNotNone(skills.find("lint", self.root))
        with mock.patch.dict(os.environ, {"LLM_SKILLS_DIRS": ""}):
            self.assertIsNone(skills.find("lint", self.root))             # hundreds of them would flood the prompt

    def test_the_folder_names_a_skill_without_a_name_field(self):
        self.project("notes", "---\ndescription: keep notes\n---\nbody")
        self.assertEqual(skills.find("notes", self.root).description, "keep notes")

    def test_a_skill_can_ask_to_be_used_by_name_only(self):
        self.project("claude-style", "---\nname: claude-style\ndescription: x\ndisable-model-invocation: true\n---\n")
        self.project("codex-style", "---\nname: codex-style\ndescription: y\n---\n",
                     extra={"agents/openai.yaml": "policy:\n  allow_implicit_invocation: false\n"})
        self.project("normal", "---\nname: normal\ndescription: z\n---\n")
        cat = skills.catalog(skills.discover(self.root))
        self.assertIn("normal", cat)
        self.assertNotIn("claude-style", cat)
        self.assertNotIn("codex-style", cat)
        self.assertIsNotNone(skills.find("codex-style", self.root))        # still usable by name


class TestWhatTheModelSees(Env):
    def test_the_catalog_stays_within_its_budget(self):
        for i in range(40):
            self.project(f"s{i:02d}", f"---\nname: s{i:02d}\ndescription: {'long words ' * 15}\n---\n")
        cat = skills.catalog(skills.discover(self.root), budget=600)
        self.assertLessEqual(len(cat.split("\n- also:")[0]), 600)
        self.assertIn("- also:", cat)
        self.assertIn("s39", cat)                                           # by name at least
        for i in range(300):
            self.project(f"many-{i:03d}", f"---\nname: many-{i:03d}\ndescription: d\n---\n")
        cat = skills.catalog(skills.discover(self.root), budget=600)
        self.assertLess(len(cat), 600 + skills.NAMES_BUDGET + 80)            # the names line is bounded too
        self.assertIn("more (use_skill finds them by name)", cat)

    def test_no_skills_no_prompt_text(self):
        self.assertEqual(skills.system_block(self.root), "")

    def test_dollar_name_brings_the_body_and_files_into_the_turn(self):
        self.project("release", "---\nname: release\ndescription: cut a release\n---\nBump, tag, notes.",
                     extra={"references/checklist.md": "x"})
        text, used = skills.expand("please $release now", self.root)
        self.assertEqual(used, ["release"])
        self.assertIn("Bump, tag, notes.", text)
        self.assertIn("references/checklist.md", text)
        self.assertTrue(text.endswith("please $release now"))

    def test_slash_name_counts_only_at_the_start(self):
        self.project("release", "---\nname: release\ndescription: r\n---\nB")
        self.assertEqual(skills.expand("/release v2", self.root)[1], ["release"])
        self.assertEqual(skills.expand("see a/release or /usr/release", self.root)[1], [])
        self.assertEqual(skills.expand("$unknown", self.root)[1], [])

    def test_a_long_body_is_cut_with_a_pointer(self):
        self.project("big", "---\nname: big\ndescription: b\n---\n" + "x" * (skills.MAX_BODY + 50))
        self.assertIn("skill cut at", skills.load(skills.find("big", self.root)))


class TestChatCommands(Env):
    def test_list_show_new_and_remove(self):
        self.assertIn("No skills yet", skills.command("/skills", self.root)[0])
        reply, follow = skills.command("/skill new triage sort incoming issues by severity", self.root)
        self.assertIsNone(follow)
        path = os.path.join(self.root, ".llm/skills/triage/SKILL.md")
        self.assertTrue(os.path.isfile(path))
        self.assertIn("triage  (project)  sort incoming issues by severity", skills.command("/skills", self.root)[0])
        self.assertIn("sort incoming issues", skills.command("/skill show triage", self.root)[0])
        skills.command("/skill rm triage", self.root)
        self.assertFalse(os.path.exists(os.path.dirname(path)))

    def test_save_hands_the_job_to_the_assistant(self):
        reply, follow = skills.command("/skill save weekly-report personal", self.root)
        self.assertIn("save_skill", follow)
        self.assertIn("personal", follow)

    def test_bad_names_and_other_tools_skills_are_refused(self):
        with self.assertRaises(skills.SkillError):
            skills.command("/skill new Bad_Name x", self.root)
        self.personal("theirs", "---\nname: theirs\ndescription: t\n---\n", where=".claude/skills")
        with self.assertRaises(skills.SkillError):
            skills.command("/skill rm theirs", self.root)
        self.assertIsNone(skills.command("hello", self.root))

    def test_personal_skills_go_to_home(self):
        path, _ = skills.save("mine", "d", "i", "personal", None)
        self.assertTrue(path.startswith(os.path.join(self.home, ".llm", "skills")))
        with self.assertRaises(skills.SkillError):
            skills.save("mine", "d", "i", "project", None)                  # no project open


class TestEditorTools(Env):
    """ui/agentloop.Tools: use_skill reads; save_skill and delete_skill go through the approval events."""

    def tools(self, answer=True, mode="manual"):
        from ui import agentloop
        from ui.server import Project
        events = []
        run = agentloop.Run()

        def emit(ev):
            events.append(ev)
            if ev.get("type") == "approval":
                threading.Timer(0.01, run.answer, ("approval:" + ev["id"], {"approve": answer})).start()
        return agentloop.Tools(Project(self.root), run, emit, mode=mode), events

    def test_save_asks_then_writes_and_use_reads_it_back(self):
        t, events = self.tools()
        out = t.call("save_skill", {"name": "deploy", "description": "ship the app", "instructions": "1. build"}, "c1")
        self.assertIn("saved skill deploy", out)
        self.assertEqual([e["kind"] for e in events if e["type"] == "approval"], ["write"])
        self.assertIn("1. build", t.call("use_skill", {"name": "deploy"}, "c2"))

    def test_a_rejected_save_writes_nothing(self):
        t, _ = self.tools(answer=False)
        self.assertIn("rejected", t.call("save_skill", {"name": "x", "description": "d", "instructions": "i"}, "c1"))
        self.assertIsNone(skills.find("x", self.root))

    def test_every_save_asks_even_in_auto_mode(self):
        # a skill joins every later system prompt: a web page must not plant one through the model unasked
        t, events = self.tools(mode="auto")
        t.call("save_skill", {"name": "p", "description": "d", "instructions": "i", "scope": "personal"}, "c1")
        t.call("save_skill", {"name": "q", "description": "d", "instructions": "i"}, "c2")
        self.assertEqual(len([e for e in events if e["type"] == "approval"]), 2)

    def test_delete_always_asks_and_plan_mode_cannot_change_skills(self):
        skills.save("old", "d", "i", "project", self.root)
        t, events = self.tools(mode="auto")
        self.assertIn("deleted", t.call("delete_skill", {"name": "old"}, "c1"))
        self.assertEqual([e["kind"] for e in events if e["type"] == "approval"], ["run"])
        plan, _ = self.tools(mode="plan")
        self.assertIn("plan mode", plan.call("save_skill", {"name": "n", "description": "d", "instructions": "i"}, "c2"))
        self.assertNotIn("save_skill", [s["function"]["name"] for s in plan.schemas()])
        self.assertIn("use_skill", [s["function"]["name"] for s in plan.schemas()])


if __name__ == "__main__":
    unittest.main()
