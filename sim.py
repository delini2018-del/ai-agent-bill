"""Agent-session cost simulator.

An AI coding agent works in turns. On every turn the model is sent the WHOLE conversation so far
(system prompt + tool schemas + every earlier message and tool result), writes a reply (billed as
output, including hidden reasoning), calls a tool, and the tool's output is appended to the history.

Token sizes of tool outputs are REAL: sampled from data/tool_output_tokens.json (real file windows, grep
results, numpy pytest runs and diffs, counted with the o200k_base tokenizer).

Prices are in "input-token units": 1 unit = the price of one uncached input token.
  output tokens     = 5 units   (e.g. Claude Haiku 5.5: $0.10 in / $0.50 out per million)
  cache read        = 0.1 units (Uber Engineering, Aug 2026, citing provider pricing)
  cache write 5-min = 1.25 units, cache write 1-hour = 2 units (same source)
"""
import json, math, random
from dataclasses import dataclass, field

TOK = json.load(open("data/tool_output_tokens.json"))
OUT_PRICE, READ, W5, W60 = 5.0, 0.1, 1.25, 2.0

@dataclass
class Policy:
    name: str
    cache_ttl: float | None = 300        # seconds; None = no prompt caching
    tool_schema: int = 50_000            # tokens of tool definitions loaded into every request (Uber: 50-70K with 100+ tools)
    subagents: bool = False              # delegate runs of reads/greps to a cheaper sub-agent
    sub_price: float = 0.2               # sub-agent model price relative to the main model
    compact_at: int | None = None        # summarise history when it passes this many tokens

SYSTEM = 12_000                          # system prompt + project instructions (assumption)

def make_session(rng, turns=None):
    """A list of turns: (action, tool_output_tokens, visible_output, reasoning, gap_seconds_after)."""
    n = turns or max(3, min(300, int(rng.lognormvariate(math.log(25), 0.8))))
    acts = []
    for _ in range(n):
        a = rng.choices(["read", "grep", "edit", "test"], [0.45, 0.20, 0.20, 0.15])[0]
        r = rng.choice(TOK[a])
        vis = int(rng.uniform(120, 500)); reason = int(rng.uniform(300, 1500))
        gap = rng.uniform(3, 15) if a != "test" else rng.uniform(20, 120)
        if rng.random() < 1 / 8:                     # the agent stops and waits for the human
            gap += rng.lognormvariate(math.log(240), 1.2)
        acts.append((a, r, vis, reason, gap))
    return acts

class Meter:
    """Bills requests with prefix caching (cache covers the previous request's whole prompt)."""
    def __init__(self, ttl, price=1.0):
        self.ttl, self.price, self.cached, self.idle, self.cost = ttl, price, 0, math.inf, 0.0
        self.parts = {"input": 0.0, "output": 0.0, "cache_read": 0.0, "new_write": 0.0, "cold_rewrite": 0.0, "uncached": 0.0}
        self.sent = 0
    def request(self, prompt_len, out_tokens):
        self.sent += prompt_len
        if self.ttl is None:
            c = prompt_len; self.parts["uncached"] += c * self.price
        else:
            w = W5 if self.ttl <= 300 else W60
            if self.cached and self.idle <= self.ttl and prompt_len >= self.cached:
                c = READ * self.cached + w * (prompt_len - self.cached)
                self.parts["cache_read"] += READ * self.cached * self.price
                self.parts["new_write"] += w * (prompt_len - self.cached) * self.price
            else:
                c = w * prompt_len                     # cold cache: the whole prompt is written again
                self.parts["cold_rewrite" if self.cached else "new_write"] += c * self.price
            self.cached = prompt_len
        self.parts["input"] += c * self.price; self.parts["output"] += OUT_PRICE * out_tokens * self.price
        self.cost += (c + OUT_PRICE * out_tokens) * self.price
        self.idle = 0.0
    def wait(self, s): self.idle += s
    def invalidate(self): self.cached = 0

def run(session, p: Policy):
    main = Meter(p.cache_ttl); sub_cost = 0.0
    prefix = SYSTEM + p.tool_schema; hist = 300       # first user message
    per_turn, peak = [], 0
    i = 0
    while i < len(session):
        a, r, vis, reason, gap = session[i]
        if p.subagents and a in ("read", "grep"):
            j = i
            while j < len(session) and session[j][0] in ("read", "grep"): j += 1
            if j - i >= 2:                             # a run of exploration -> one sub-agent call
                sm = Meter(300, p.sub_price); sctx = 4_000 + 300
                for k in range(i, j):
                    _, rk, vk, rsk, gk = session[k]
                    sm.request(sctx, vk + rsk); sctx += vk + rk; sm.wait(min(gk, 15))
                sm.request(sctx, 400)                  # sub-agent writes a summary
                sub_cost += sm.cost
                before = main.cost
                main.request(prefix + hist, vis + reason)   # main agent: delegate, then read the summary
                hist += vis + 400; main.wait(gap)
                per_turn.append((main.cost - before) + sm.cost); i = j; peak = max(peak, prefix + hist)
                continue
        before = main.cost
        main.request(prefix + hist, vis + reason)
        hist += vis + r; main.wait(gap)
        if p.compact_at and hist > p.compact_at:       # summarise the history (one extra call)
            main.request(prefix + hist, 3_000); hist = 8_000; main.invalidate()
        per_turn.append(main.cost - before); peak = max(peak, prefix + hist); i += 1
    return dict(cost=main.cost + sub_cost, turns=len(session), per_turn=per_turn, peak=peak,
                input=main.parts["input"], output=main.parts["output"], sub=sub_cost, parts=main.parts, sent=main.sent)

POLICIES = [
    Policy("No caching", cache_ttl=None),
    Policy("Default: 5-min cache, all tools loaded"),
    Policy("1-hour cache", cache_ttl=3600),
    Policy("Load tools on demand", tool_schema=5_000),
    Policy("Compact history at 100K", compact_at=100_000),
    Policy("Cheaper sub-agents for exploration", subagents=True),
    Policy("All four fixes", cache_ttl=3600, tool_schema=5_000, subagents=True, compact_at=100_000),
]
