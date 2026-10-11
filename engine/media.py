"""Pictures sent to a model that cannot read them: dropped with a note, never an error.

A client (the editor, any OpenAI-style app) may attach an image to a model that has no vision tower — or has
one switched off. The request still deserves an answer, so the image parts are taken out and one line tells
the model (and so the user) what happened. Stdlib only."""

IMAGE_TYPES = ("image_url", "image", "input_image")
NOTE = "[{n} image{s} omitted: this model cannot read images]"


def is_image(part):
    return isinstance(part, dict) and (part.get("type") in IMAGE_TYPES
                                       or (part.get("type") is None and ("image_url" in part or "image" in part)))


def strip_images(messages):
    """(messages without image parts, number of images dropped). The input is not modified; a message that
    lost images gets a closing text part saying so."""
    out, dropped = [], 0
    for m in messages:
        c = m.get("content") if isinstance(m, dict) else None
        if not isinstance(c, list) or not any(is_image(p) for p in c):
            out.append(m)
            continue
        n = sum(1 for p in c if is_image(p))
        dropped += n
        keep = [p for p in c if not is_image(p)]
        keep.append({"type": "text", "text": ("\n" if keep else "") + NOTE.format(n=n, s="" if n == 1 else "s")})
        out.append({**m, "content": keep})
    return out, dropped
