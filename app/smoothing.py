"""Milestone 2 (part 2): smoothing methods, all sharing one interface over the same `NGramCounts`.

    model.prob(word, history)      -> P(word | history)         (Stupid Backoff: a score, not a probability)
    model.distribution(history)    -> {word: P} over every predictable token

A model of order N conditions on the last N-1 tokens of `history` (left-padded with <s> when shorter).
`distribution` sums to 1.0 for every method except Stupid Backoff.
"""
from app.corpus import BOS
from app.ngram import NGramCounts


class LanguageModel:
    name = "base"
    normalised = True        # False for scoring schemes such as Stupid Backoff

    def __init__(self, counts: NGramCounts):
        self.counts = counts
        self.order = counts.order
        self.V = len(counts.predictable)

    def prob(self, word: str, history: list[str]) -> float:
        raise NotImplementedError

    def distribution(self, history: list[str]) -> dict[str, float]:
        return {w: self.prob(w, history) for w in self.counts.predictable}

    def params(self) -> dict:
        return {}


class AddK(LanguageModel):
    """Add-k (Laplace when k=1): pretend every possible continuation was seen k extra times. No backoff."""
    name = "add-k"

    def __init__(self, counts, k: float = 0.1):
        super().__init__(counts)
        self.k = k

    def prob(self, word, history):
        ctx = self.counts.pad_history(history)
        return (self.counts.count(ctx, word) + self.k) / (self.counts.context_total(ctx) + self.k * self.V)

    def params(self):
        return {"k": self.k}


class JelinekMercer(LanguageModel):
    """Interpolated Jelinek-Mercer: P_m = lam * ML_m + (1 - lam) * P_(m-1), bottoming out in a uniform distribution."""
    name = "jelinek-mercer"

    def __init__(self, counts, lam: float = 0.7):
        super().__init__(counts)
        self.lam = lam

    def prob(self, word, history):
        h = self.counts.pad_history(history)
        c = self.counts
        p = self.lam * c.count((), word) / c.context_total(()) + (1 - self.lam) / self.V
        for m in range(2, self.order + 1):
            ctx = h[len(h) - (m - 1):]
            tot = c.context_total(ctx)
            if tot:                                  # unseen context: nothing to interpolate, keep lower order
                p = self.lam * c.count(ctx, word) / tot + (1 - self.lam) * p
        return p

    def params(self):
        return {"lam": self.lam}


class StupidBackoff(LanguageModel):
    """Stupid Backoff (Brants et al. 2007): relative frequency if seen, else alpha * the lower-order score.
    The scores do NOT sum to 1, so this is a ranking score and has no valid perplexity."""
    name = "stupid-backoff"
    normalised = False

    def __init__(self, counts, alpha: float = 0.4):
        super().__init__(counts)
        self.alpha = alpha

    def prob(self, word, history):
        h = self.counts.pad_history(history)
        scale = 1.0
        for m in range(self.order, 1, -1):
            ctx = h[len(h) - (m - 1):]
            c = self.counts.count(ctx, word)
            if c:
                return scale * c / self.counts.context_total(ctx)
            scale *= self.alpha
        return scale * self.counts.count((), word) / self.counts.context_total(())

    def params(self):
        return {"alpha": self.alpha}


class KneserNey(LanguageModel):
    """Interpolated Kneser-Ney with absolute discounting D.

    Highest order:  P(w|h) = max(c(hw) - D, 0)/c(h) + D * N1+(h.)/c(h) * P_lower(w|h')
    Lower orders use *continuation counts*  N1+(. h w) = number of distinct words that precede h w,  instead of raw
    counts, so a word that is frequent but only ever follows one word ("francisco") is not over-predicted after
    unrelated contexts. N-grams that start with <s> keep their raw counts (nothing can precede them).
    The unigram level is interpolated with the uniform distribution, so every word keeps non-zero probability.
    """
    name = "kneser-ney"

    def __init__(self, counts, discount: float = 0.75):
        super().__init__(counts)
        if not 0 < discount < 1:
            raise ValueError("discount must be in (0, 1)")
        self.D = discount
        self._build_continuation_tables()

    def _build_continuation_tables(self):
        c, n = self.counts, self.order
        self.levels: dict[int, dict[tuple, tuple]] = {}      # m -> ctx -> (row, total)
        for m in range(1, n + 1):
            if m == n:
                rows = c.tables[m]
            else:
                rows = {}
                for ctx_hi, row in c.tables[m + 1].items():  # every distinct (m+1)-gram adds 1 to its m-gram tail
                    tail = ctx_hi[1:]
                    out = rows.setdefault(tail, {})
                    for w in row:
                        out[w] = out.get(w, 0) + 1
                for ctx, row in c.tables[m].items():         # <s>-initial n-grams: raw counts
                    if ctx and ctx[0] == BOS:
                        rows[ctx] = dict(row)
            self.levels[m] = {ctx: (row, sum(row.values())) for ctx, row in rows.items()}

    def prob(self, word, history):
        h = self.counts.pad_history(history)
        D = self.D
        row, tot = self.levels[1][()]
        p = max(row.get(word, 0) - D, 0) / tot + D * len(row) / tot / self.V
        for m in range(2, self.order + 1):
            entry = self.levels[m].get(h[len(h) - (m - 1):])
            if entry is None:
                continue
            row, tot = entry
            p = max(row.get(word, 0) - D, 0) / tot + D * len(row) / tot * p
        return p

    def params(self):
        return {"discount": self.D}


METHODS = {"add-k": AddK, "jelinek-mercer": JelinekMercer, "stupid-backoff": StupidBackoff, "kneser-ney": KneserNey}
