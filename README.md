# The AI Bill Doesn't Grow. It Compounds.

Code for the Medium article. A coding-agent cost simulator fed with REAL tool outputs (Python stdlib file
windows, grep results, numpy pytest runs, diffs) counted with the o200k_base tokenizer, run on 2,000 sessions
under seven billing setups.

| Result | Number |
|---|---|
| Tokens sent by a median 24-step session | ~1.9 million (for ~45K tokens of new material) |
| No caching vs 5-minute cache | 4.2× the bill |
| Share of default bill spent re-sending history after the cache expired | 28% |
| Load tools on demand / cheaper sub-agents / 1-hour cache / compaction | −31% / −44% / −15% / −8% |
| All four fixes | −67% (Uber reported −52% cost per session) |
| 1/5-price model writing 3× output: break-even | ~2.4× the steps |

Prices in input-token units: output 5, cache read 0.1, 5-min write 1.25, 1-h write 2 (see article for sources).

```bash
pip install -r requirements.txt
sh run_all.sh
```
`data/tool_output_tokens.json` holds the measured token counts, so `build_corpus.py` (needs Node + numpy tests +
pytest) is optional. Or open `colab_notebook.ipynb`. Licence: MIT.
