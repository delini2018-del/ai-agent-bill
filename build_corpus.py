"""Build a library of REAL tool outputs a coding agent would see, and count their tokens with a real
tokenizer (OpenAI o200k_base via the gpt-tokenizer npm package).
  read  : a 40-250 line window of a real Python standard-library source file
  grep  : real `grep -rn` output over the standard library for a common identifier (capped at 60 lines)
  test  : real output of running pytest on one of numpy's own test modules
  edit  : a real unified diff between two versions of a stdlib file window (a few lines changed)
"""
import json, os, random, subprocess, sysconfig, difflib, glob
rng = random.Random(0)
LIB = sysconfig.get_paths()["stdlib"]
files = [f for f in glob.glob(LIB + "/**/*.py", recursive=True) if "/test" not in f and os.path.getsize(f) > 4000]
rng.shuffle(files)
texts = {"read": [], "grep": [], "test": [], "edit": []}
for f in files[:400]:
    lines = open(f, errors="ignore").read().splitlines()
    n = rng.randint(40, 250); s = rng.randint(0, max(0, len(lines) - n))
    texts["read"].append("\n".join(f"{i+1}\t{l}" for i, l in enumerate(lines[s:s+n], start=s)))
    w = lines[s:s+30]
    if len(w) > 10:
        new = list(w); k = rng.randrange(len(new)); new[k] = new[k] + "  # changed"; new.insert(k, "    # TODO: handle edge case")
        texts["edit"].append("\n".join(difflib.unified_diff(w, new, f, f, lineterm="")))
words = ["self", "return", "def ", "import", "raise", "None", "yield", "async", "lambda", "except", "config", "parse",
         "encode", "buffer", "socket", "thread", "lock", "timeout", "path", "header"]
for wd in words * 5:
    out = subprocess.run(["grep", "-rn", "--include=*.py", wd, LIB + "/" + rng.choice(["json", "email", "http", "asyncio", "logging", "xml", "urllib", "concurrent", "importlib", "unittest"])],
                         capture_output=True, text=True).stdout.splitlines()
    rng.shuffle(out); texts["grep"].append("\n".join(out[:rng.randint(5, 60)]))
for mod in ["lib.tests.test_arraypad", "lib.tests.test_arraysetops", "lib.tests.test_function_base", "lib.tests.test_shape_base",
            "lib.tests.test_twodim_base", "lib.tests.test_index_tricks", "lib.tests.test_nanfunctions", "lib.tests.test_polynomial",
            "linalg.tests.test_linalg", "fft.tests.test_pocketfft", "polynomial.tests.test_chebyshev", "ma.tests.test_extras",
            "lib.tests.test_histograms", "lib.tests.test_stride_tricks", "random.tests.test_generator_mt19937"]:
    # real pytest runs of numpy's own test modules (verbose, as an agent would run them)
    r = subprocess.run(["python3", "-m", "pytest", "--pyargs", "numpy." + mod, "-v", "-x", "-p", "no:cacheprovider"],
                       capture_output=True, text=True, timeout=600)
    out = (r.stdout + r.stderr).splitlines()
    for k in range(4):   # several slices, as an agent sees truncated/partial runs
        n = rng.randint(20, 200); s = rng.randint(0, max(0, len(out) - n)); texts["test"].append("\n".join(out[s:s+n]))
counts = {}
for k, v in texts.items():
    v = [t for t in v if t.strip()]
    c = json.loads(subprocess.run(["node", "tok/count.mjs"], input=json.dumps(v), capture_output=True, text=True).stdout)
    counts[k] = c
json.dump(counts, open("data/tool_output_tokens.json", "w"))
for k, c in counts.items():
    c = sorted(c); print(f"{k:5s} n={len(c):4d} median={c[len(c)//2]:6d} p90={c[int(len(c)*.9)]:6d} max={c[-1]}")
