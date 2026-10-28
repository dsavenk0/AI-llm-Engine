# Perception: turn whatever the user attaches into text the Conductor can route on.
#
# This is the glue that makes the orchestrator multimodal and fully on-device: an attachment is recognised
# here (audio → Whisper, a scan or image or PDF → OCR, a document → text extraction) and the result becomes a
# "source" the planner sees and each subtask can draw the relevant passages from. So the user attaches a scan,
# a recording or a spec, and the senior/code model works on the recognised content — no cloud, no manual step.
import os

from engine import audio, documents


def recognize(paths):
    """Recognise attachments into sources: [{"name", "kind", "text", "note"}], in the given order. Audio goes
    through Whisper, everything else through the document extractor (which itself OCRs scans and images). A
    file that cannot be read becomes a source with a note instead of stopping the run."""
    out = []
    for p in paths:
        full = os.path.expanduser(p)
        name = os.path.basename(full.rstrip("/")) or full
        ext = os.path.splitext(full)[1].lower()
        try:
            if ext in audio.AUDIO_EXTS:
                r = audio.transcribe(full)
                out.append({"name": name, "kind": "audio", "text": r.get("text", ""), "note": ""})
            else:
                r = documents.extract_text(full)
                out.append({"name": name, "kind": r["kind"], "text": r["text"], "note": r.get("note", "")})
        except (documents.DocError, audio.AudioError) as e:
            out.append({"name": name, "kind": "error", "text": "", "note": str(e)})
    return out


def as_text(sources):
    """The recognised sources as one text block (headers + body or note), for a planner prompt or a log."""
    parts = []
    for s in sources:
        body = s.get("text") or ("[" + (s.get("note") or "unreadable") + "]")
        parts.append(f"### {s.get('name', 'source')} ({s.get('kind', 'file')})\n{body}")
    return "\n\n".join(parts)
