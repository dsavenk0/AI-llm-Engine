"""Where does a decode step actually go?

Splits one generated token into:
  floor    — weights / measured memory bandwidth (what the memory bus alone costs)
  forward  — the model's own forward pass, mx.eval'd (GPU work + kernel launches)
  outside  — everything else per token in the engine loop (sampling, stop filter, detokenise, Python)

Run:  .venv/bin/python -m tests.profile <model-path>
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import mlx.core as mx                                           # noqa: E402
from engine.generate import Engine                              # noqa: E402
from engine.model import Positions                              # noqa: E402


def bandwidth_gbs():
    """Measured sequential read bandwidth, so the floor uses this machine's real number."""
    n = int(1.0 * 2**30 / 2)
    x = mx.random.normal((n,)).astype(mx.float16)
    mx.eval(x)
    for _ in range(3):
        mx.eval(mx.sum(x))
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        mx.eval(mx.sum(x))
        ts.append(time.perf_counter() - t)
    return x.nbytes / min(ts) / 1e9


def main(path):
    bw = bandwidth_gbs()
    eng = Engine(path, vision=False, max_context=8192, max_tokens=512, ptq1=True, log=lambda *a: None)
    model = eng.model
    wb = eng.model_bytes

    # --- raw forward, one token, 512-token context ---
    cache = model.make_cache()
    mx.eval(model(mx.array([list(range(1000, 1512))]), cache))

    def step(n, pipelined):
        """Per-forward cost. `pipelined` mimics the engine's decode loop (async_eval, CPU overlaps GPU);
        the synchronous variant is the naive one-eval-per-step cost, for comparison."""
        out = None
        t = time.perf_counter()
        for _ in range(n):
            out = model(mx.array([[1000]]), cache, Positions(), last_only=False)
            if pipelined:
                mx.async_eval(out)
            else:
                mx.eval(out)
            for c in cache:
                if hasattr(c, "trim_to"):
                    c.trim_to(512)
        if out is not None:
            mx.eval(out)
        return (time.perf_counter() - t) / n

    step(5, True)                                               # warm-up
    fwd = step(25, True)                                        # pipelined, like the real loop
    fwd_sync = step(25, False)                                  # naive synchronous, for comparison

    # --- full engine decode, speculation off so one token == one forward ---
    eng.speculative = False
    eng.cache = None
    msgs = [{"role": "user", "content": "Write a detailed story about a bonsai garden in Kyoto."}]
    text, _, _ = eng.build_prompt(msgs, thinking=False)
    ids, _ = eng.encode(text, [])

    def run(maxt):
        eng.cache = None
        for item in eng.generate(ids, sampling={"temperature": 0}, max_tokens=maxt):
            if isinstance(item, dict):
                return item
        return {}

    run(16)                                                     # warm-up
    st = max((run(200) for _ in range(2)), key=lambda s: s.get("decode_tps", 0))
    per_tok = st["decode_s"] / max(1, st["completion_tokens"])

    floor = wb / (bw * 1e9)
    outside = per_tok - fwd
    gpu_over = fwd - floor

    print(f"\n=== {os.path.basename(path.rstrip('/'))} ===")
    print(f"weights {wb / 2**30:.2f} GB · measured bandwidth {bw:.0f} GB/s · decode {st['decode_tps']:.1f} tok/s"
          f" (no speculation)")
    print(f"{'per token (engine)':<24} {per_tok * 1e3:7.2f} ms   100.0 %")
    print(f"{'  weight-read floor':<24} {floor * 1e3:7.2f} ms   {floor / per_tok * 100:5.1f} %")
    print(f"{'  GPU work above floor':<24} {gpu_over * 1e3:7.2f} ms   {gpu_over / per_tok * 100:5.1f} %")
    print(f"{'  CPU/Python outside GPU':<24} {outside * 1e3:7.2f} ms   {outside / per_tok * 100:5.1f} %")
    print(f"{'forward, pipelined':<24} {fwd * 1e3:7.2f} ms")
    print(f"{'forward, naive sync':<24} {fwd_sync * 1e3:7.2f} ms   "
          f"(+{(fwd_sync - fwd) / fwd * 100:.0f} % — what the pipelined loop already saves)")
    print(f"prefill {st.get('prefill_tps', 0):.0f} tok/s on {st.get('prompt_tokens', 0)} prompt tokens")


if __name__ == "__main__":
    main(os.path.expanduser(sys.argv[1]))
