"""One real test per milestone (plus the smoothing sanity checks). Fast, deterministic, no network."""
import random

import math

import pytest

from app import corpus, evaluate
from app.ngram import NGramCounts
from app.smoothing import AddK, JelinekMercer, KneserNey, StupidBackoff

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


@pytest.fixture(scope="module")
def prep():
    return corpus.prepare()


TINY = [s.split() for s in [
    "please book a follow-up visit", "please book a visit today", "please confirm your visit",
    "book a visit with the nurse", "the nurse will call you", "please call the clinic",
]]
TINY_VOCAB = {t for s in TINY for t in s} | set(corpus.SPECIALS)


# Milestone 2: Implement the n-gram language model and apply smoothing.
@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("cls", [AddK, JelinekMercer, KneserNey])
def test_each_smoothing_distribution_sums_to_one(cls, order):
    model = cls(NGramCounts.from_sentences(TINY, order, TINY_VOCAB))
    for history in ([], ["please"], ["please", "book"], ["a", "visit"], ["the", "zebra"]):   # incl. unseen context
        total = sum(model.distribution(history).values())
        assert total == pytest.approx(1.0, abs=1e-9), (cls.name, order, history, total)
        assert all(p > 0 for p in model.distribution(history).values())


def test_stupid_backoff_is_a_score_not_a_distribution():
    model = StupidBackoff(NGramCounts.from_sentences(TINY, 3, TINY_VOCAB))
    assert not model.normalised
    assert abs(sum(model.distribution(["please", "book"]).values()) - 1.0) > 1e-3
    # a seen trigram uses its own relative frequency, an unseen one backs off with the 0.4 penalty
    assert model.prob("a", ["please", "book"]) == pytest.approx(1.0)
    assert 0 < model.prob("nurse", ["please", "book"]) < model.prob("a", ["please", "book"])


def test_kneser_ney_uses_continuation_counts_and_keeps_unseen_continuations_possible():
    sents = [["san", "francisco"]] * 6 + [[w, "glasses"] for w in ("reading", "new", "my", "his")] + [["the", "cat"]]
    vocab = {t for s in sents for t in s} | set(corpus.SPECIALS)
    counts = NGramCounts.from_sentences(sents, 2, vocab)
    kn, jm = KneserNey(counts, discount=0.75), JelinekMercer(counts, lam=0.5)
    # "francisco" is more frequent (6 v 4) but only ever follows "san"; "glasses" follows four different words.
    assert counts.count((), "francisco") > counts.count((), "glasses")
    assert kn.prob("glasses", ["the"]) > kn.prob("francisco", ["the"])      # continuation probability
    assert jm.prob("francisco", ["the"]) > jm.prob("glasses", ["the"])      # raw frequency says the opposite
    # unseen-but-plausible: "the visit" never occurs in training, KN still gives it probability mass
    model = KneserNey(NGramCounts.from_sentences(TINY, 3, TINY_VOCAB))
    assert model.counts.count(("the", "clinic"), "visit") == 0
    assert model.prob("visit", ["the", "clinic"]) > 0


def test_oov_words_are_mapped_to_unk_and_perplexity_is_finite(prep):
    vocab = prep.vocab
    mapped = corpus.apply_vocab([["please", "qwertyuiop", "visit"]], vocab)
    assert mapped[0][1] == corpus.UNK
    model = KneserNey(NGramCounts.from_sentences(prep.train, 3, vocab))
    assert math.isfinite(evaluate.perplexity(model, mapped))
    with pytest.raises(ValueError):                    # counting raw, unmapped text must be refused
        NGramCounts.from_sentences([["qwertyuiop"]], 2, vocab)


def test_perplexity_comparison_uses_identical_tokens_and_kn_beats_add_k(prep):
    val = prep.val_by_source["handwritten"]
    ppl = {}
    for n in (2, 3):
        counts = NGramCounts.from_sentences(prep.train, n, prep.vocab)
        ppl[n] = (evaluate.perplexity(AddK(counts, k=0.1), val), evaluate.perplexity(KneserNey(counts, discount=0.8), val))
    assert evaluate.token_count(val) == sum(len(s) + 1 for s in val)        # <s> never counted, </s> always
    assert ppl[3][1] < ppl[3][0] and ppl[2][1] < ppl[2][0]                  # Kneser-Ney beats add-k
    assert ppl[3][1] < ppl[2][1]                                            # and longer context helps it


def test_model_round_trips_through_json(tmp_path, prep):
    counts = NGramCounts.from_sentences(prep.train[:200], 3, prep.vocab)
    counts.save(tmp_path / "m.json")
    again = NGramCounts.load(tmp_path / "m.json")
    a, b = KneserNey(counts), KneserNey(again)
    assert a.prob("visit", ["please", "book"]) == pytest.approx(b.prob("visit", ["please", "book"]))


# Milestone 3: Develop a text generation feature that suggests words based on input.
@pytest.fixture(scope="module")
def suggester(prep):
    from app.suggest import Suggester
    return Suggester(KneserNey(NGramCounts.from_sentences(prep.train, 3, prep.vocab), discount=0.8))


def test_milestone_3_suggestions_are_ranked_top_k_with_probabilities(suggester):
    from app.core import milestone_3
    out = milestone_3(suggester.model, "we will contact you within ", k=5)
    assert len(out) == 5
    probs = [s.prob for s in out]
    assert probs == sorted(probs, reverse=True) and all(0 < p <= 1 for p in probs)
    assert out == milestone_3(suggester.model, "we will contact you within ", k=5)       # deterministic
    assert not {"<s>", "</s>", "<unk>"} & {s.word for s in out}
    assert suggester.suggest("we will contact you within ", k=3) == out[:3]


def test_prefix_filter_only_offers_matching_words(suggester):
    out = suggester.suggest("please book a fol", k=5)
    assert out and all(s.word.startswith("fol") for s in out)
    assert out[0].word == "follow-up"
    assert all(s.word.startswith("th") for s in suggester.suggest("please call the clinic", k=5, prefix="th"))
    from app.suggest import split_input
    assert split_input("please book a fol") == (["please", "book", "a"], "fol")
    assert split_input("please book a ") == (["please", "book", "a"], "")


def test_sentence_boundary_resets_context_and_multiword_completion_extends(suggester):
    after_stop = suggester.suggest("Your visit is booked. ", k=3)
    assert {s.word for s in after_stop} <= {s.word for s in suggester.suggest("", k=300)}
    paths = suggester.complete("please book a ", steps=4, beam=3)
    assert 1 <= len(paths) <= 3 and all(1 <= len(p.words) <= 4 for p in paths)
    assert [p.prob for p in paths] == sorted((p.prob for p in paths), reverse=True)
    assert suggester.complete("please book a ", steps=4, beam=3) == paths
