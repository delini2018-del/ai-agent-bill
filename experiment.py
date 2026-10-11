"""Run every policy on the same 2,000 simulated sessions, plus the 'growth' and 'cheap model' studies."""
import json, math, random, statistics as st
import numpy as np
from sim import make_session, run, POLICIES, Policy

rng = random.Random(42)
sessions = [make_session(rng) for _ in range(2000)]
R = {"policies": {}}
for p in POLICIES:
    res = [run(s, p) for s in sessions]
    R["policies"][p.name] = dict(total=sum(r["cost"] for r in res), mean=st.mean(r["cost"] for r in res),
                                 output_share=sum(r["output"] for r in res) / sum(r["cost"] for r in res))
# Where the default bill goes
p = POLICIES[1]; parts = {}
for s_ in sessions:
    for k, v in run(s_, p)["parts"].items(): parts[k] = parts.get(k, 0) + v
tot = sum(v for k, v in parts.items() if k in ("cache_read", "new_write", "cold_rewrite", "output"))
R["default_breakdown"] = {k: parts[k] / tot for k in ("cache_read", "new_write", "cold_rewrite", "output")}
# Tokens sent (what the model actually reads) vs session length, no caching discount: exponent for >= 100 turns
xs = [len(s_) for s_ in sessions]; sent = [run(s_, POLICIES[0])["sent"] for s_ in sessions]
big = [(math.log(x), math.log(y)) for x, y in zip(xs, sent) if x >= 100]
R["sent_exponent_100plus"] = float(np.polyfit([a for a, _ in big], [b for _, b in big], 1)[0])
R["sent_median_tokens"] = float(np.median(sent)); R["turns_median"] = float(np.median(xs))
base = R["policies"]["Default: 5-min cache, all tools loaded"]["total"]
for k, v in R["policies"].items(): v["vs_default"] = v["total"] / base - 1

# Growth: one fixed sequence of actions, cut at different lengths (same turns, just longer sessions)
long = make_session(random.Random(7), turns=300)
R["growth"] = {}
for name in ["No caching", "Default: 5-min cache, all tools loaded"]:
    p = next(x for x in POLICIES if x.name == name)
    full = run(long, p)["per_turn"]
    R["growth"][name] = dict(per_turn=full, cumulative=list(np.cumsum(full)))
# Scaling exponent across the 2,000 sessions: cost ~ turns^k
for name in ["No caching", "Default: 5-min cache, all tools loaded"]:
    p = next(x for x in POLICIES if x.name == name)
    xs = [len(s) for s in sessions]; ys = [run(s, p)["cost"] for s in sessions]
    big = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x >= 40]
    k = np.polyfit([a for a, _ in big], [b for _, b in big], 1)[0]
    R["growth"][name]["exponent_40plus"] = float(k)
    R["growth"][name]["scatter"] = list(zip(xs, ys))
# Share of the bill spent in the last quarter of a session (sessions with >= 40 turns, default policy)
p = POLICIES[1]; shares = []
for s in sessions:
    if len(s) >= 40:
        pt = run(s, p)["per_turn"]; q = int(len(pt) * 0.75); shares.append(sum(pt[q:]) / sum(pt))
R["last_quarter_share"] = st.mean(shares)
pt = R["growth"]["Default: 5-min cache, all tools loaded"]["per_turn"]
R["turn_ratio_100_vs_10"] = float(np.mean(pt[95:105]) / np.mean(pt[5:15]))
cum = R["growth"]["Default: 5-min cache, all tools loaded"]["cumulative"]
R["cost_ratio_100_vs_50_turns"] = float(cum[99] / cum[49]); R["cost_ratio_200_vs_100_turns"] = float(cum[199] / cum[99])

# Cheap per token vs cheap per task: model B costs `price` x model A per token, but needs `m` x the turns
# (more steps / retries) and writes `v` x the output tokens.
R["cheap"] = {}
base_sessions = sessions[:400]
costA = sum(run(s, POLICIES[1])["cost"] for s in base_sessions)
for price in [1/3, 1/5, 1/10]:
    row = []
    for m in [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
        tot = 0.0; r2 = random.Random(int(m * 100))
        for s in base_sessions:
            n = max(3, round(len(s) * m))
            ext = [s[i % len(s)] for i in range(n)]                    # same kinds of steps, more of them
            ext = [(a, r, v * 3, rs * 3, g) for a, r, v, rs, g in ext]   # ~3x the output tokens (Artificial Analysis)
            tot += run(ext, POLICIES[1])["cost"] * price
        row.append((m, tot / costA))
    R["cheap"][f"{price:.3f}"] = row
# Sensitivity: how idle humans turn the cache cold (default 5-min cache vs 1-hour)
R["idle"] = {}
for med in [30, 60, 120, 240, 480, 900]:
    rr = random.Random(5); ss = []
    for s_ in sessions[:500]:
        ss.append([(a, r, v, rs, (g if g < 200 else rr.lognormvariate(math.log(med), 1.2))) for a, r, v, rs, g in s_])
    row = {}
    for p in (POLICIES[1], POLICIES[2]):
        res = [run(x, p) for x in ss]
        row[p.name] = sum(r["cost"] for r in res)
        row[p.name + " cold share"] = sum(r["parts"]["cold_rewrite"] for r in res) / sum(r["cost"] for r in res)
    R["idle"][med] = row
json.dump(R, open("results/results.json", "w"))
s = {k: v for k, v in R.items() if k not in ("growth", "cheap")}
s["exponents"] = {k: v["exponent_40plus"] for k, v in R["growth"].items()}; s["cheap"] = R["cheap"]
json.dump(s, open("results/summary.json", "w"), indent=1)
print(json.dumps(s, indent=1))
