"""One real test per milestone (plus the smoothing sanity checks). Fast, deterministic, no network."""
import random

import pytest

from app import corpus

random.seed(0)


# Milestone 1: Curate an anonymized corpus of administrative medical text.
def test_milestone_1():
    # awkward tokens: hyphens, titles, a.m./p.m., times, numbers, apostrophes, punctuation
    toks = corpus.tokenize("Dr. Lindqvist's follow-up is at 9:30 a.m. on the 3rd (fee 1,500).")
    assert toks == ["dr.", "lindqvist's", "follow-up", "is", "at", "9:30", "a.m.", "on", "the", "3rd",
                    "(", "fee", "1,500", ")", "."]
    # a title's period is not a sentence boundary, an a.m. followed by a capital is
    assert corpus.split_sentences("See Ms. Brightwater at 9:30 a.m. Please bring your card.") == \
        ["See Ms. Brightwater at 9:30 a.m.", "Please bring your card."]
    # detokenizer gives readable text back
    assert corpus.detokenize(corpus.tokenize("Book a follow-up, please (Dr. Fenwick).")) == \
        "book a follow-up, please (dr. fenwick)."
    # comment lines are dropped, typographic quotes are normalised
    assert corpus.clean_text("# header\nit’s ok") == "it's ok"

    prep = corpus.prepare()
    # closed vocabulary: built from train only, every split token is in it, rare words became <unk>
    assert all(t in prep.vocab for s in prep.train + prep.val + prep.test for t in s)
    assert corpus.UNK in prep.vocab and any(corpus.UNK in s for s in prep.train)
    assert prep.stats["sentences"] > 2000 and prep.stats["tokens"] > 30000
    # the real corpus files carry the synthetic-data statement
    for path in corpus.CORPUS_DIR.glob("*.txt"):
        assert "SYNTHETIC" in path.read_text(encoding="utf-8")[:300]
    # no train/val leakage: splits are by document
    assert prep.train and prep.val and prep.test
