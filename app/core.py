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


# --- Milestone 2: Implement the n-gram language model and apply smoothing.
def milestone_2(prepared: corpus.Prepared, order: int = 4, method: str = "kneser-ney", **params):
    """Count n-grams on the training split and wrap them in the chosen smoothing method."""
    from app.ngram import NGramCounts
    from app.smoothing import METHODS
    return METHODS[method](NGramCounts.from_sentences(prepared.train, order, prepared.vocab), **params)


# --- Milestone 3: Develop a text generation feature that suggests words based on input.
def milestone_3(model, text: str, k: int = 5, prefix: str | None = None, steps: int = 0, beam: int = 3):
    """Top-k next-word suggestions (with probabilities) for `text`, honouring a typed prefix.

    With steps > 0 returns multi-word completions (deterministic beam search) instead.
    """
    from app.suggest import Suggester
    suggester = Suggester(model)
    if steps:
        return suggester.complete(text, steps=steps, beam=beam)
    return suggester.suggest(text, k=k, prefix=prefix)


def train_full_model(corpus_dir: Path | str = corpus.CORPUS_DIR, order: int = 4, method: str = "kneser-ney",
                     min_count: int = 2, **params):
    """Train on the WHOLE corpus (no held-out split) for use in the CLI. Returns (model, corpus stats)."""
    from app.ngram import NGramCounts
    from app.smoothing import METHODS
    docs = corpus.load_documents(corpus_dir)
    sentences = corpus.sentences_of(docs)
    vocab = corpus.build_vocab(sentences, min_count)
    counts = NGramCounts.from_sentences(corpus.apply_vocab(sentences, vocab), order, vocab)
    return METHODS[method](counts, **params), corpus.corpus_stats(docs)


# --- Milestone 4: Create a CLI for users to interact with the suggestion engine.
def milestone_4(argv: list[str] | None = None) -> None:
    """The command-line interface lives in app/cli.py (train, suggest, evaluate and the interactive draft REPL)."""
    from app.cli import main
    main(argv)
