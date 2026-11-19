"""Milestone 3: turn a language model into next-word suggestions.

Deterministic autocomplete, not creative generation: candidates are ranked by probability (ties broken
alphabetically) and nothing is sampled, so the same input always gives the same list.
"""
import math
import re
from dataclasses import dataclass

from app.corpus import BOS, EOS, SPECIALS, tokenize
from app.smoothing import LanguageModel

SENTENCE_END = {".", "?", "!"}
_WORDLIKE = re.compile(r"[a-z0-9]", re.I)


@dataclass(frozen=True)
class Suggestion:
    word: str
    prob: float


@dataclass(frozen=True)
class Completion:
    words: tuple
    prob: float            # joint probability of the whole extension


def split_input(text: str) -> tuple[list[str], str]:
    """Split typed text into (context tokens, prefix being typed).

    "please book a fol"  -> (["please", "book", "a"], "fol")      the last word is still being typed
    "please book a "     -> (["please", "book", "a"], "")         the word is finished
    "see you at 9:30 a.m." -> (..., "")                            a token ending in punctuation is finished
    """
    tokens = tokenize(text)
    if tokens and text and not text[-1].isspace() and _WORDLIKE.match(tokens[-1][0]) and tokens[-1][-1] not in ".:":
        return tokens[:-1], tokens[-1]
    return tokens, ""


def sentence_history(tokens: list[str]) -> list[str]:
    """Only tokens since the last sentence end are context (the model was trained with <s> at each start)."""
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i] in SENTENCE_END:
            return tokens[i + 1:]
    return tokens


class Suggester:
    def __init__(self, model: LanguageModel):
        self.model = model

    def _candidates(self, history: list[str], prefix: str, allow_punct: bool) -> list[Suggestion]:
        dist = self.model.distribution(history)
        out = []
        for word, p in dist.items():
            if word in SPECIALS:
                continue
            if prefix:
                if not word.startswith(prefix) or not _WORDLIKE.match(word[0]):
                    continue
            elif not allow_punct and not _WORDLIKE.match(word[0]):
                continue
            out.append(Suggestion(word, p))
        out.sort(key=lambda s: (-s.prob, s.word))
        return out

    def suggest(self, text: str, k: int = 5, prefix: str | None = None, allow_punct: bool = True) -> list[Suggestion]:
        """Top-k next words with their model probabilities P(word | context).

        If `prefix` is None it is taken from the unfinished last word of `text`; with a prefix only words that
        start with it are offered. The probabilities are the model's, not renormalised over the filtered list.
        """
        tokens, typed_prefix = split_input(text)
        prefix = typed_prefix if prefix is None else prefix.lower()
        return self._candidates(sentence_history(tokens), prefix, allow_punct)[:k]

    def complete(self, text: str, steps: int = 4, beam: int = 3) -> list[Completion]:
        """Extend the text by up to `steps` words with a deterministic beam search; best joint probability first.

        A path stops early when it emits sentence-ending punctuation. A half-typed word is completed first.
        """
        tokens, prefix = split_input(text)
        beams = [((), 0.0, False)]          # (new words, log prob, finished)
        for step in range(steps):
            nxt = []
            for words, logp, done in beams:
                if done:
                    nxt.append((words, logp, True))
                    continue
                history = sentence_history(tokens + list(words))
                for s in self._candidates(history, prefix if step == 0 else "", allow_punct=step > 0)[:beam]:
                    nxt.append((words + (s.word,), logp + math.log(s.prob), s.word in SENTENCE_END))
            beams = sorted(nxt, key=lambda b: (-b[1], b[0]))[:beam]
        return [Completion(w, math.exp(lp)) for w, lp, _ in beams if w]
