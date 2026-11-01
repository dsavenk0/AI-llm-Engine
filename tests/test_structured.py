"""The JSON grammar behind constrained decoding (engine/structured.py).

Two kinds of test, and the second kind is the one that matters. The first asks "is this document valid?",
which any JSON parser answers. The second asks "can the model still get out of here?" — because a
constraint that only rejects the wrong *answer* lets the model walk into a corner it cannot leave, and then
it loops until the token budget runs out. Every bug found by running a real model was of the second kind:
an enum checked only when the string closed, a comma allowed when no key could follow it, a token that
decodes to nothing passing every check forever.

    .venv/bin/python -m unittest tests.test_structured -v
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine.structured import Cursor, finished, supported  # noqa: E402

PERSON = {"type": "object",
          "properties": {"name": {"type": "string"},
                         "age": {"type": "integer"},
                         "tags": {"type": "array", "items": {"type": "string"}},
                         "mood": {"enum": ["good", "bad"]}},
          "required": ["name", "age"]}


class TestPlainJson(unittest.TestCase):
    """Without a schema it is exactly JSON: nothing less, nothing more."""

    def test_accepts_every_shape_of_document(self):
        for t in ('{}', '[]', '{"a":1}', '[1,2,3]', '"hi"', '42', '-0.5', '1.5e-3', 'true', 'false', 'null',
                  '{"a":[1,{"b":null}]}', '{"":"empty key"}', '[[]]', '  {"a" : 1 } ',
                  '{"a":"\\u00e9"}', '{"a":"tab\\there"}'):
            self.assertTrue(finished(t), t)

    def test_rejects_what_is_not_json(self):
        for t in ('{', '{"a"}', '{"a":}', '[1,]', '{"a":1,}', '{,}', '"unterminated', '{}{}', 'tru',
                  '{"a":1}x', '[1 2]', '{"a":1"b":2}', "{'a':1}", '{"a":01}'):
            self.assertFalse(finished(t), t)

    def test_control_characters_must_be_escaped(self):
        self.assertFalse(finished('{"a":"x\x16y"}'))
        self.assertFalse(finished('{"a":"\x00"}'))
        self.assertFalse(finished('{"a":"line\nbreak"}'))
        self.assertTrue(finished('{"a":"x\\u0016y"}'))

    def test_the_root_kind_can_be_pinned(self):
        self.assertTrue(finished('{"a":1}', root="object"))
        self.assertFalse(finished('[1]', root="object"))
        self.assertFalse(finished('42', root="object"))      # a scalar never opens a container


class TestSchema(unittest.TestCase):
    """What the schema subset actually enforces."""

    def test_accepts_documents_the_schema_allows(self):
        for t in ('{"name":"a","age":3}',
                  '{"name":"a","age":3,"tags":["x","y"]}',
                  '{"name":"a","age":3,"mood":"good"}',
                  '{"name":"a","age":-7,"tags":[]}'):
            self.assertTrue(finished(t, PERSON), t)

    def test_rejects_documents_the_schema_forbids(self):
        for t, why in (('{"name":"a"}', "age is required"),
                       ('{"age":3}', "name is required"),
                       ('{"name":1,"age":3}', "name is not a string"),
                       ('{"name":"a","age":3.5}', "age is not an integer"),
                       ('{"name":"a","age":3,"zz":1}', "zz is not in the schema"),
                       ('{"name":"a","age":3,"tags":[1]}', "tags holds strings"),
                       ('{"name":"a","age":3,"mood":"meh"}', "mood is not in the enum"),
                       ('{"name":"a","age":3,"name":"b"}', "duplicate key")):
            self.assertFalse(finished(t, PERSON), why)

    def test_says_up_front_what_it_will_not_enforce(self):
        self.assertIsNone(supported(PERSON))
        for schema in ({"type": "object", "patternProperties": {"^a": {}}},
                       {"type": "object", "properties": {"a": {"minimum": 3}}},
                       {"type": "fancy"}):
            self.assertIsNotNone(supported(schema), schema)


class TestNoDeadEnds(unittest.TestCase):
    """The constraint must refuse a token that leaves the model with no legal continuation.

    Each of these was a real bug: the document was correctly judged invalid at the end, but the model had
    already been allowed into a state from which nothing could finish it.
    """

    def test_an_enum_is_checked_while_it_is_typed(self):
        c = Cursor(PERSON, "object")
        c.feed('{"name":"a","age":3,"mood": "')
        self.assertTrue(c.accepts("go"))
        self.assertFalse(c.accepts("me"))          # "meh" could never close legally

    def test_a_key_is_checked_while_it_is_typed(self):
        self.assertTrue(Cursor(PERSON, "object").accepts('{"na'))
        self.assertFalse(Cursor(PERSON, "object").accepts('{"zz'))

    def test_a_key_cannot_be_spelled_with_escapes(self):
        self.assertFalse(Cursor(PERSON, "object").accepts('{"\\u006eame"'))

    def test_a_comma_needs_a_key_to_follow_it(self):
        both = {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a"]}
        c = Cursor(both, "object")
        c.feed('{"a":1,"b":2')
        self.assertFalse(c.accepts(","))           # nothing could come after it
        self.assertTrue(c.accepts("}"))
        c2 = Cursor(both, "object")
        c2.feed('{"a":1')
        self.assertTrue(c2.accepts(","))           # b is still free

    def test_an_object_cannot_close_while_something_is_required(self):
        c = Cursor(PERSON, "object")
        c.feed('{"name":"a"')
        self.assertFalse(c.accepts("}"))
        c.feed(',"age":3')
        self.assertTrue(c.accepts("}"))


class TestCompletion(unittest.TestCase):
    """When stopping is legal — the engine ends the reply the moment the document is whole."""

    def test_a_document_is_complete_only_once_it_closes(self):
        c = Cursor(PERSON, "object")
        c.feed('{"name":"a","age":3')
        self.assertFalse(c.complete())
        c.feed('}')
        self.assertTrue(c.complete())

    def test_a_number_at_the_root_has_no_closing_character(self):
        c = Cursor()
        c.feed("42")
        self.assertTrue(c.complete())              # nothing follows it, and nothing needs to
        c2 = Cursor()
        c2.feed("-")
        self.assertFalse(c2.complete())

    def test_nothing_may_follow_a_finished_document(self):
        c = Cursor()
        c.feed('{"a":1}')
        self.assertTrue(c.accepts("  "))           # trailing whitespace is fine
        self.assertFalse(c.accepts("x"))


if __name__ == "__main__":
    unittest.main()
