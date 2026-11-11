"""A node's capability document (docs/adar/PROTOCOL.md) — every field labelled by how it was obtained:
measured, static, estimated, or unavailable. Built from numbers the engine already computes for itself
(engine/memory.py), never from new hardware detection: chip generation and thermal state stay
"unavailable" because no code anywhere in this repo reads them, and guessing is exactly what the labelling
exists to prevent.

A simulated node's document is built from the arguments it's given, with `"simulated": true` at the top —
it never reports this Mac's real memory as if it were another machine's."""
import platform
import shutil

from .protocol import HANDLERS, VERSION


def _f(value, source):
    return {"value": value, "source": source}


def build(identity, *, models=(), handlers=HANDLERS, active_tasks=0, queue_depth=0, simulated=False,
          memory_gb=None, available_gb=None, pressure="normal", loaded_models=()):
    if simulated:
        static_mem = _f(memory_gb, "static") if memory_gb is not None else _f(None, "unavailable")
        dyn_mem = _f(available_gb, "measured") if available_gb is not None else _f(None, "unavailable")
        dyn_pressure = _f(pressure, "measured")
        disk = _f(None, "unavailable")
        macos = _f(None, "unavailable")
    else:
        from engine import memory
        static_mem = _f(round(memory.TOTAL / 2**30, 1), "static")
        dyn_mem = _f(round(memory.available_bytes() / 2**30, 1), "measured")
        dyn_pressure = _f(memory.pressure_level(), "measured")
        try:
            disk = _f(round(shutil.disk_usage("/").free / 2**30, 1), "measured")
        except OSError:
            disk = _f(None, "unavailable")
        macos = _f(platform.mac_ver()[0] or None, "static" if platform.mac_ver()[0] else "unavailable")
    return {
        "v": VERSION,
        "node_id": identity.node_id,
        "simulated": bool(simulated),
        "static": {
            "unified_memory_gb": static_mem,
            "macos_version": macos,
            "chip_generation": _f(None, "unavailable"),
            "adar_version": _f("0.2", "static"),
            "installed_models": _f(sorted(models), "static"),
            "task_handlers": _f(sorted(handlers), "static"),
        },
        "dynamic": {
            "available_memory_gb": dyn_mem,
            "memory_pressure": dyn_pressure,
            "loaded_models": _f(sorted(loaded_models), "measured"),   # warm: no load cost to run on
            "active_tasks": _f(active_tasks, "measured"),
            "queue_depth": _f(queue_depth, "measured"),
            "thermal_state": _f(None, "unavailable"),
            "disk_free_gb": disk,
        },
    }


def value(doc, section, field):
    return doc[section][field]["value"]


def check(doc):
    """A capability document whose own claims don't hang together is refused before anything is routed by
    it — a handler the protocol does not define, a model loaded but not installed, a value marked
    unavailable — so the contradiction is caught here and not when a task fails on that node."""
    from .protocol import ProtocolError, check_version
    check_version(doc, what="capability document")
    for section in ("static", "dynamic"):
        for name, field in doc.get(section, {}).items():
            if not isinstance(field, dict) or field.get("source") not in ("measured", "static", "estimated",
                                                                        "unavailable"):
                raise ProtocolError(f"capability field {section}.{name} is not labelled measured/static/"
                                    f"estimated/unavailable")
            if field["source"] == "unavailable" and field.get("value") is not None:
                raise ProtocolError(f"capability field {section}.{name} says 'unavailable' but carries a value")
    unknown = set(value(doc, "static", "task_handlers")) - set(HANDLERS)
    if unknown:
        raise ProtocolError(f"node claims handlers this protocol version doesn't define: {sorted(unknown)}")
    loaded = doc.get("dynamic", {}).get("loaded_models", {}).get("value") or []
    ghost = set(loaded) - set(value(doc, "static", "installed_models"))
    if ghost:
        raise ProtocolError(f"node claims to have loaded models it does not list as installed: {sorted(ghost)}")
    return doc
