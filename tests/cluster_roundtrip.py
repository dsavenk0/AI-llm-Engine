"""Does splitting the layers across two ranks give the same answer as one process?

This is the test that keeps the cluster work from being written blind. A second Mac is not needed for it:
MLX's ring backend runs both ranks on one machine over loopback, so the *correctness* of the split is
provable here. What is not provable here is the benefit — both ranks share this machine's RAM, so nothing
larger fits — and that claim is not made until it runs on two Macs.

    .venv/bin/python tests/cluster_roundtrip.py

It drives itself: with no rank in the environment it builds a synthetic bundle, starts two children over
loopback, and compares their logits against the ordinary single-process path.
"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PROMPT = [list(range(100, 116))]


def child():
    """One rank: load the model, keep only this rank's layers' caches, run the pipelined forward."""
    import mlx.core as mx
    from engine import cluster
    from engine.loader import load

    group = cluster.init(backend="ring")
    if group is None:
        print("RANK ERROR: no group", flush=True)
        return 1
    bundle = os.environ["BUNDLE"]
    with open(os.path.join(bundle, "config.json")) as f:
        config = json.load(f)
    lo, hi = cluster.my_layers(config, group)
    # only this rank's layers are read off disk — the memory half of the split
    model, _, nbytes = load(bundle, vision=False, log=lambda *a: None, layer_range=(lo, hi))
    print(f"RANKINFO rank={group.rank()} layers={lo}..{hi} loaded={nbytes}", flush=True)
    cache = model.make_cache()                     # one entry per owned layer, by construction
    logits = cluster.forward(model, cache, group, ids=mx.array(PROMPT))
    mx.eval(logits)
    if group.rank() == 0:
        out = [float(x) for x in logits[0, -1, :16].tolist()]
        print("RANK0 " + json.dumps({"shape": list(logits.shape), "head": out}), flush=True)
    return 0


def reference(bundle):
    """The same prompt through the ordinary single-process path."""
    import mlx.core as mx
    from engine.loader import load
    model, _, _ = load(bundle, vision=False, log=lambda *a: None)
    logits = model(mx.array(PROMPT), model.make_cache())
    mx.eval(logits)
    return list(logits.shape), [float(x) for x in logits[0, -1, :16].tolist()]


def main():
    if os.environ.get("MLX_RANK") is not None:
        return child()

    from tests.synth import CHATML, make_bundle
    bundle = tempfile.mkdtemp()
    make_bundle(bundle, CHATML, seed=3)
    want_shape, want_head = reference(bundle)
    print(f"single process: logits {want_shape}")

    d = tempfile.mkdtemp()
    hosts = os.path.join(d, "hosts.json")
    with open(hosts, "w") as f:
        json.dump(["127.0.0.1:19101", "127.0.0.1:19102"], f)

    env = dict(os.environ, MLX_HOSTFILE=hosts, BUNDLE=bundle)
    procs = [subprocess.Popen([sys.executable, "-B", __file__], cwd=ROOT,
                              env=dict(env, MLX_RANK=str(r)),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
             for r in (1, 0)]
    outs = [p.communicate(timeout=300)[0] for p in procs]
    for r, o in zip((1, 0), outs):
        for line in o.splitlines():
            if line.startswith("RANKINFO") or "Error" in line or "error" in line:
                print(f"  rank {r}: {line}")

    got = next((l for o in outs for l in o.splitlines() if l.startswith("RANK0 ")), None)
    if not got:
        print("FAILED: rank 0 produced no logits")
        return 1
    data = json.loads(got[len("RANK0 "):])
    if data["shape"] != want_shape:
        print(f"FAILED: shape {data['shape']} against {want_shape}")
        return 1
    worst = max(abs(a - b) for a, b in zip(data["head"], want_head))
    print(f"two ranks over loopback: logits {data['shape']}, largest difference {worst:.6f}")
    ok = worst < 1e-3
    print("OK — the split answers the same as one process" if ok else f"FAILED: differs by {worst}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
