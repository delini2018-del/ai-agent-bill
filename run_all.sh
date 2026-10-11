#!/bin/sh
set -e
# Optional: rebuild the real tool-output token library (needs Node and `npm install` in tok/)
# (cd tok && npm install) && python build_corpus.py
python experiment.py
python make_charts.py
