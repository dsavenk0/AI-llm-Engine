import unittest

from engine.media import is_image, strip_images

IMG = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}


class TestStripImages(unittest.TestCase):
    def test_images_go_and_a_note_says_how_many(self):
        msgs = [{"role": "system", "content": "be brief"},
                {"role": "user", "content": [{"type": "text", "text": "what is this?"}, IMG, IMG]}]
        out, n = strip_images(msgs)
        self.assertEqual(n, 2)
        self.assertEqual(out[0], msgs[0])
        parts = out[1]["content"]
        self.assertEqual(parts[0], {"type": "text", "text": "what is this?"})
        self.assertEqual(len(parts), 2)
        self.assertIn("2 images omitted", parts[1]["text"])
        self.assertEqual(len(msgs[1]["content"]), 3)                  # the input is left alone

    def test_a_single_image_and_an_image_only_message(self):
        out, n = strip_images([{"role": "user", "content": [IMG]}])
        self.assertEqual(n, 1)
        self.assertEqual(out[0]["content"], [{"type": "text", "text": "[1 image omitted: this model cannot read images]"}])

    def test_plain_messages_and_odd_shapes_pass_through(self):
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": None, "tool_calls": []},
                {"role": "user", "content": [{"type": "text", "text": "x"}]}, "not a dict"]
        out, n = strip_images(msgs)
        self.assertEqual((out, n), (msgs, 0))

    def test_every_spelling_of_an_image_part_is_recognised(self):
        for part in ({"type": "image_url", "image_url": {"url": "u"}}, {"type": "image", "image": "u"},
                     {"type": "input_image", "image_url": "u"}, {"image_url": {"url": "u"}}, {"image": "u"}):
            self.assertTrue(is_image(part), part)
        for part in ({"type": "text", "text": "x"}, {}, "x", None, {"type": "text", "image": "u"}):
            self.assertFalse(is_image(part), part)


if __name__ == "__main__":
    unittest.main()
