"""Charts for the article, from results/results.json."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = json.load(open("results/results.json")); os.makedirs("charts", exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
INK, RED, AMB, BLUE, GREY, GREEN = "#1f2937", "#c2410c", "#e9a23b", "#2563eb", "#94a3b8", "#15803d"
K = 1e6   # show costs in "millions of input-token units" -> label as $ at $1 per million input tokens

# 1 ── one 300-turn session: the cost of each turn
g = R["growth"]
fig, ax = plt.subplots(figsize=(11, 5.4))
nc = np.array(g["No caching"]["per_turn"]) / K; dc = np.array(g["Default: 5-min cache, all tools loaded"]["per_turn"]) / K
t = np.arange(1, len(nc) + 1)
ax.plot(t, nc, color=GREY, lw=1.6, label="No prompt caching")
ax.plot(t, dc, color=BLUE, lw=1.6, label="5-minute prompt cache (the usual default)")
cold = np.where((t > 2) & (dc > np.convolve(dc, np.ones(9) / 9, mode="same") * 2.2))[0]
ax.scatter(t[cold], dc[cold], color=RED, s=28, zorder=3, label="Cache went cold: the whole conversation is re-sent at full price")
ax.set_xlabel("Turn number in one coding-agent session"); ax.set_ylabel("Cost of that single turn ($)")
ax.legend(frameon=False, loc="upper left", fontsize=11)
fig.suptitle("Every turn re-sends everything, so each turn costs more than the last", x=0.02, ha="left", fontsize=14, color=INK)
fig.text(0.02, 0.005, r"Illustrative price: \$1 per million input tokens, \$5 per million output tokens.", fontsize=9.5, color="#64748b")
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig("charts/01_per_turn.png", dpi=200); plt.close(fig)

# 2 ── where the default bill goes + what each fix saves
pol = R["policies"]; order = ["1-hour cache", "Compact history at 100K", "Load tools on demand",
                              "Cheaper sub-agents for exploration", "All four fixes"]
lab = {"1-hour cache": "1-hour cache instead of 5-minute", "Compact history at 100K": "Summarise history at 100K tokens",
       "Load tools on demand": "Load tool definitions only when needed", "Cheaper sub-agents for exploration":
       "Cheaper sub-agents do the reading", "All four fixes": "All four together"}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [1, 1.6]})
b = R["default_breakdown"]; parts = [("Re-reading cached history\n(at 10% price)", b["cache_read"], BLUE),
     ("Re-sending the WHOLE history\nafter the cache went cold", b["cold_rewrite"], RED),
     ("Writing new tokens\ninto the cache", b["new_write"], AMB), ("Model output\n(incl. reasoning)", b["output"], GREY)]
y = 0
for name, v, c in parts:
    a1.bar(0, v, bottom=y, color=c, width=0.6); a1.text(0.36, y + v / 2, f"{v*100:.0f}%  {name}", va="center", fontsize=10.5)
    y += v
a1.set_xlim(-0.4, 2.4); a1.set_ylim(0, 1); a1.axis("off"); a1.set_title("Where the default bill goes", loc="left", fontsize=12.5)
vals = [pol[k]["vs_default"] * 100 for k in order]
a2.barh(range(len(order)), vals, color=[GREEN if k == "All four fixes" else BLUE for k in order])
for i, v in enumerate(vals): a2.text(v + 1.2, i, f"{v:.0f}%", va="center", ha="left", color="white", fontsize=11.5, fontweight="bold")
a2.axvline(-52, color=RED, ls="--", lw=1.4); a2.text(-51, -0.62, "Uber reported −52% cost per session", color=RED, fontsize=10, ha="left", va="bottom")
a2.set_yticks(range(len(order)), [lab[k] for k in order]); a2.set_ylim(len(order) - 0.5, -0.9); a2.set_xlim(-75, 0)
a2.set_xlabel("Change in total cost vs the default (%)"); a2.set_title("What each fix saves (same 2,000 sessions)", loc="left", fontsize=12.5)
fig.tight_layout(); fig.savefig("charts/02_bill_and_fixes.png", dpi=200); plt.close(fig)

# 3 ── idle humans and the cache
idle = R["idle"]; xs = sorted(int(k) for k in idle)
fig, ax = plt.subplots(figsize=(11, 5))
d5 = [idle[str(x)]["Default: 5-min cache, all tools loaded"] for x in xs]; d60 = [idle[str(x)]["1-hour cache"] for x in xs]
ref = d5[0]
ax.plot([x / 60 for x in xs], [v / ref for v in d5], color=BLUE, lw=2.6, marker="o", label="5-minute cache")
ax.plot([x / 60 for x in xs], [v / ref for v in d60], color=AMB, lw=2.6, marker="o", label="1-hour cache (writes cost 2× instead of 1.25×)")
ax.set_xscale("log"); ax.set_xticks([0.5, 1, 2, 4, 8, 15], ["30 s", "1 min", "2 min", "4 min", "8 min", "15 min"])
ax.set_xlabel("Typical time a human takes to reply when the agent stops and asks (median)")
ax.set_ylabel("Total cost (relative)"); ax.legend(frameon=False)
fig.suptitle("Fast humans should keep the short cache. Slow humans pay for every coffee break.", x=0.02, ha="left", fontsize=14, color=INK)
fig.tight_layout(); fig.savefig("charts/03_idle.png", dpi=200); plt.close(fig)

# 4 ── cheap per token vs cheap per task
fig, ax = plt.subplots(figsize=(11, 5.2))
for key, c, l in [("0.333", AMB, "1/3 the price per token"), ("0.200", RED, "1/5 the price per token"), ("0.100", BLUE, "1/10 the price per token")]:
    m, r = zip(*R["cheap"][key]); ax.plot(m, r, color=c, lw=2.6, marker="o", label=l)
ax.axhline(1, color=INK, lw=1); ax.text(1.02, 1.04, "same cost as the expensive model", fontsize=10.5)
ax.set_xlabel("How many more turns the cheap model needs to finish the same task (×)")
ax.set_ylabel("Cheap model's bill ÷ expensive model's bill"); ax.legend(frameon=False, loc="upper left")
fig.suptitle("A cheaper model that also writes 3× the tokens stops being cheap fast", x=0.02, ha="left", fontsize=14, color=INK)
fig.tight_layout(); fig.savefig("charts/04_cheap_model.png", dpi=200); plt.close(fig)
print("ok")
