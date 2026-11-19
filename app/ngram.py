"""Milestone 2 (part 1): n-gram counting.

`NGramCounts` holds counts for every order 1..N as  tables[m][context_tuple][word] = count,  where the context has
m-1 tokens. Sentences are padded with N-1 `<s>` tokens on the left and one `</s>` on the right; every token after
the padding (words and `</s>`, never `<s>`) is a prediction target exactly once per order, so all orders see the
same events and `<s>` is only ever context.
"""
import json
from collections import defaultdict
from pathlib import Path

from app.corpus import BOS, EOS, SPECIALS


class NGramCounts:
    def __init__(self, order: int, vocab: set[str]):
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.vocab = set(vocab) | set(SPECIALS)
        # tokens the model can predict: everything except the start-of-sentence marker
        self.predictable = sorted(self.vocab - {BOS})
        self.tables: dict[int, dict[tuple, dict[str, int]]] = {m: defaultdict(dict) for m in range(1, order + 1)}
        self.totals: dict[int, dict[tuple, int]] = {m: {} for m in range(1, order + 1)}

    # ---- training -------------------------------------------------------------------------------
    def add_sentences(self, sentences: list[list[str]]) -> "NGramCounts":
        n = self.order
        for sent in sentences:
            bad = [t for t in sent if t not in self.vocab]
            if bad:
                raise ValueError(f"token outside the closed vocabulary (map to <unk> first): {bad[0]!r}")
            padded = [BOS] * (n - 1) + list(sent) + [EOS]
            for i in range(n - 1, len(padded)):
                word = padded[i]
                for m in range(1, n + 1):
                    ctx = tuple(padded[i - m + 1:i])
                    row = self.tables[m][ctx]
                    row[word] = row.get(word, 0) + 1
        self._recompute_totals()
        return self

    def _recompute_totals(self) -> None:
        for m, table in self.tables.items():
            self.totals[m] = {ctx: sum(row.values()) for ctx, row in table.items()}

    @classmethod
    def from_sentences(cls, sentences, order: int, vocab: set[str]) -> "NGramCounts":
        return cls(order, vocab).add_sentences(sentences)

    # ---- queries --------------------------------------------------------------------------------
    def count(self, ctx: tuple, word: str) -> int:
        return self.tables[len(ctx) + 1].get(ctx, {}).get(word, 0)

    def context_total(self, ctx: tuple) -> int:
        return self.totals[len(ctx) + 1].get(ctx, 0)

    def pad_history(self, history: list[str]) -> tuple:
        """Left-pad with <s> / truncate so the history has exactly order-1 tokens, as at training time."""
        n = self.order - 1
        h = list(history)[-n:] if n else []
        return tuple([BOS] * (n - len(h)) + h)

    # ---- persistence ----------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "order": self.order,
            "vocab": sorted(self.vocab),
            "tables": {str(m): {" ".join(ctx): row for ctx, row in table.items()} for m, table in self.tables.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NGramCounts":
        obj = cls(d["order"], set(d["vocab"]))
        for m, table in d["tables"].items():
            for ctx, row in table.items():
                obj.tables[int(m)][tuple(ctx.split(" ")) if ctx else ()] = dict(row)
        obj._recompute_totals()
        return obj

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "NGramCounts":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
