"""One function per milestone. Each is a thin orchestrator; the real work lives in focused modules:

    app/corpus.py     milestone 1  cleaning, tokenizing, splitting, closed vocabulary
    app/ngram.py      milestone 2  n-gram counts
    app/smoothing.py  milestone 2  add-k, Jelinek-Mercer, stupid backoff, interpolated Kneser-Ney
    app/suggest.py    milestone 3  ranked top-k suggestions, prefix filter, multi-word completion
    app/cli.py        milestone 4  train / suggest / evaluate / draft commands
"""
from pathlib import Path

from app import corpus


# --- Milestone 1: Curate an anonymized corpus of administrative medical text.
def milestone_1(corpus_dir: Path | str = corpus.CORPUS_DIR, min_count: int = 2, seed: int = 13,
                sources: tuple = ("handwritten", "generated")) -> corpus.Prepared:
    """Load the synthetic corpus, clean + tokenize it, split by document and build the closed vocabulary."""
    return corpus.prepare(corpus_dir, min_count=min_count, seed=seed, sources=sources)
