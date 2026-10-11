"""A small mutation tester: does the suite notice when the code is changed?

    .venv/bin/python tests/mutate.py engine/steering.py tests.test_steering [--max 40] [--seed 1] [-k pattern]

Each mutant changes one thing (a comparison, an arithmetic operator, and/or, a constant, a condition) in a COPY
of the repository, runs the given unittest targets there, and reports whether a test failed ("killed") or
everything still passed ("survived" — a hole in the tests, or an equivalent mutant). The working tree is never
touched. Exit code 1 when anything survived, so it can gate a change.
"""
import ast
import copy
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {".venv", ".venv-images", ".git", "llama.cpp", "__pycache__", "LLMEditor.app", "LLMBar.app", "LLMVoice.app",
             "dist", "node_modules", ".llm", "third_party"}

CMP_SWAP = {ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
            ast.In: ast.NotIn, ast.NotIn: ast.In, ast.Is: ast.IsNot, ast.IsNot: ast.Is}
BIN_SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv, ast.FloorDiv: ast.Mult, ast.Div: ast.Mult}


class Collector(ast.NodeVisitor):
    """Lists (kind, lineno, description, apply(tree_copy_node)) for every place a mutation makes sense."""

    def __init__(self):
        self.sites = []

    def add(self, node, what):
        self.sites.append((getattr(node, "lineno", 0), what, node))

    def visit_Compare(self, n):
        if len(n.ops) == 1 and type(n.ops[0]) in CMP_SWAP:
            self.add(n, f"compare {type(n.ops[0]).__name__} → {CMP_SWAP[type(n.ops[0])].__name__}")
        self.generic_visit(n)

    def visit_BinOp(self, n):
        if type(n.op) in BIN_SWAP:
            self.add(n, f"operator {type(n.op).__name__} → {BIN_SWAP[type(n.op)].__name__}")
        self.generic_visit(n)

    def visit_BoolOp(self, n):
        self.add(n, f"{'and' if isinstance(n.op, ast.And) else 'or'} → {'or' if isinstance(n.op, ast.And) else 'and'}")
        self.generic_visit(n)

    def visit_If(self, n):
        self.add(n, "if condition negated")
        self.generic_visit(n)

    def visit_While(self, n):
        self.generic_visit(n)

    def visit_Constant(self, n):
        if isinstance(n.value, bool) or (isinstance(n.value, int) and not isinstance(n.value, bool)):
            self.add(n, f"constant {n.value!r}")
        self.generic_visit(n)

    def visit_Return(self, n):
        if n.value is not None and not (isinstance(n.value, ast.Constant) and n.value.value is None):
            self.add(n, "return value → None")
        self.generic_visit(n)

    def visit_AugAssign(self, n):
        if type(n.op) in (ast.Add, ast.Sub):
            self.add(n, f"augmented {type(n.op).__name__} flipped")
        self.generic_visit(n)


def apply(node):
    if isinstance(node, ast.Compare):
        node.ops[0] = CMP_SWAP[type(node.ops[0])]()
    elif isinstance(node, ast.BinOp):
        node.op = BIN_SWAP[type(node.op)]()
    elif isinstance(node, ast.BoolOp):
        node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
    elif isinstance(node, ast.If):
        node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
    elif isinstance(node, ast.Constant):
        node.value = (not node.value) if isinstance(node.value, bool) else node.value + 1
    elif isinstance(node, ast.Return):
        node.value = ast.Constant(value=None)
    elif isinstance(node, ast.AugAssign):
        node.op = ast.Sub() if isinstance(node.op, ast.Add) else ast.Add()


def make_copy():
    tmp = tempfile.mkdtemp(prefix="mutate-")
    for name in os.listdir(ROOT):
        if name in SKIP_DIRS or name.endswith(".app"):
            continue
        src = os.path.join(ROOT, name)
        dst = os.path.join(tmp, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".venv*"))
        else:
            shutil.copy2(src, dst)
    return tmp


def run_tests(cwd, targets, patterns, timeout):
    cmd = [sys.executable, "-B", "-m", "unittest", "-q", *targets]
    for p in patterns:
        cmd += ["-k", p]
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", LLM_HOME=cwd))
    except subprocess.TimeoutExpired:
        return "timeout"                       # a mutant that hangs is noticed too
    return "pass" if p.returncode == 0 else "fail"


def main(argv):
    args = [a for a in argv if not a.startswith("--") and a != "-k"]
    opts = {}
    patterns, it = [], iter(range(len(argv)))
    for i in it:
        if argv[i] == "--max":
            opts["max"] = int(argv[i + 1]); next(it)
        elif argv[i] == "--seed":
            opts["seed"] = int(argv[i + 1]); next(it)
        elif argv[i] == "--timeout":
            opts["timeout"] = int(argv[i + 1]); next(it)
        elif argv[i] == "--lines":
            lo, hi = argv[i + 1].split("-"); opts["lines"] = (int(lo), int(hi)); next(it)
        elif argv[i] == "-k":
            patterns.append(argv[i + 1]); next(it)
    skip = {str(v) for k, v in opts.items() if k != "lines"} | {f"{opts['lines'][0]}-{opts['lines'][1]}"} \
        if "lines" in opts else {str(v) for v in opts.values()}
    args = [a for a in args if a not in skip and a not in patterns]
    if len(args) < 2:
        print(__doc__); return 2
    target, tests = args[0], args[1:]
    path = os.path.join(ROOT, target)
    with open(path, encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source)
    col = Collector()
    col.visit(tree)
    natural = list(col.sites)                      # walk order: the same in every fresh parse
    lo, hi = opts.get("lines", (0, 10 ** 9))
    pool = [x for x in natural if lo <= x[0] <= hi]
    random.Random(opts.get("seed", 1)).shuffle(pool)
    sites = sorted(pool[:opts.get("max", 40)], key=lambda s: s[0])
    work = make_copy()
    base = run_tests(work, tests, patterns, opts.get("timeout", 300))
    if base != "pass":
        print(f"the unmodified code already fails these tests ({base}); fix that first")
        shutil.rmtree(work, ignore_errors=True)
        return 2
    print(f"{target}: {len(col.sites)} mutation sites, trying {len(sites)}; tests: {' '.join(tests)}")
    survivors, killed = [], 0
    dst = os.path.join(work, target)
    try:
        for line, what, node in sites:
            fresh = ast.parse(source)
            c2 = Collector()
            c2.visit(fresh)
            # the same site in the fresh tree: same position in a deterministic walk
            idx = [id(s[2]) for s in natural].index(id(node))
            _, _, twin = c2.sites[idx]
            apply(twin)
            ast.fix_missing_locations(fresh)
            with open(dst, "w", encoding="utf-8") as f:
                f.write(ast.unparse(fresh))
            t0 = time.time()
            res = run_tests(work, tests, patterns, opts.get("timeout", 300))
            if res == "pass":
                survivors.append((line, what))
                print(f"  SURVIVED  line {line}: {what}   ({time.time() - t0:.0f}s)")
            else:
                killed += 1
    finally:
        shutil.rmtree(work, ignore_errors=True)
    total = killed + len(survivors)
    print(f"{target}: {killed}/{total} killed" + ("" if not survivors else f", {len(survivors)} survived"))
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
