"""Held-out evaluation: perplexity grid over orders 1-5 and smoothing methods, plus next-word accuracy.

Rules that keep the numbers comparable:
  * every model uses the SAME closed vocabulary (built from training data, rare words -> <unk>);
  * every model is scored on the SAME validation tokens;
  * each sentence is padded with <s> (not counted) and </s> (counted), so the token total never depends on n.
"""
import math

from app.corpus import BOS, EOS
from app.ngram import NGramCounts
from app.smoothing import AddK, JelinekMercer, KneserNey, StupidBackoff

ORDERS = (1, 2, 3, 4, 5)
PARAM_GRID = {
    "add-k": [("k", v) for v in (0.001, 0.01, 0.05, 0.1, 0.5, 1.0)],              # 1.0 is Laplace
    "jelinek-mercer": [("lam", v) for v in (0.2, 0.4, 0.6, 0.8, 0.9)],
    "kneser-ney": [("discount", v) for v in (0.5, 0.7, 0.8, 0.9, 0.95)],
}
CLASSES = {"add-k": AddK, "jelinek-mercer": JelinekMercer, "kneser-ney": KneserNey}


def token_count(sentences) -> int:
    """Scored tokens: every word plus one </s> per sentence. <s> padding is never counted."""
    return sum(len(s) + 1 for s in sentences)


def perplexity(model, sentences) -> float:
    n = model.order
    log_sum = 0.0
    for sent in sentences:
        padded = [BOS] * (n - 1) + list(sent) + [EOS]
        for i in range(n - 1, len(padded)):
            p = model.prob(padded[i], padded[max(0, i - (n - 1)):i])
            log_sum += math.log(p)
    return math.exp(-log_sum / token_count(sentences))


def top_k_accuracy(model, sentences, ks=(1, 3), max_positions=400) -> dict:
    """Share of positions where the true next token is in the model's top k (works for any scoring scheme)."""
    n = model.order
    positions = []
    for sent in sentences:
        padded = [BOS] * (n - 1) + list(sent) + [EOS]
        positions += [(padded[max(0, i - (n - 1)):i], padded[i]) for i in range(n - 1, len(padded))]
    step = max(1, len(positions) // max_positions)
    positions = positions[::step][:max_positions]
    hits = {k: 0 for k in ks}
    for hist, truth in positions:
        scores = model.distribution(hist)
        ranked = sorted(scores, key=lambda w: (-scores[w], w))
        for k in ks:
            hits[k] += truth in ranked[:k]
    return {k: hits[k] / len(positions) for k in ks}


def build_counts(train, vocab, orders=ORDERS) -> dict:
    return {n: NGramCounts.from_sentences(train, n, vocab) for n in orders}


def tune_grid(counts_by_order: dict, tune_sentences, report_sets: dict) -> dict:
    """For each method and order choose the hyper-parameter with the lowest perplexity on `tune_sentences`, then
    report perplexity of that choice on every set in `report_sets`. Returns results[method][n]."""
    results = {}
    for method, grid in PARAM_GRID.items():
        results[method] = {}
        for n, counts in counts_by_order.items():
            best = None
            for name, value in grid:
                model = CLASSES[method](counts, **{name: value})
                ppl = perplexity(model, tune_sentences)
                if best is None or ppl < best[0]:
                    best = (ppl, name, value)
            _, name, value = best
            model = CLASSES[method](counts, **{name: value})
            results[method][n] = {"param": (name, value),
                                  "ppl": {label: perplexity(model, sents) for label, sents in report_sets.items()}}
    return results


def best_cell(results: dict, label: str):
    """(ppl, method, n) with the lowest perplexity on report set `label`."""
    return min((cell["ppl"][label], method, n) for method, by_n in results.items() for n, cell in by_n.items())


def format_ppl_table(results: dict, label: str) -> str:
    methods = list(results)
    head = "| n | " + " | ".join(methods) + " |"
    sep = "|---|" + "|".join("---" for _ in methods) + "|"
    best = best_cell(results, label)
    rows = []
    for n in sorted(next(iter(results.values()))):
        cells = []
        for method in methods:
            cell = results[method][n]
            text = f"{cell['ppl'][label]:.1f} ({cell['param'][0]}={cell['param'][1]:g})"
            if (method, n) == (best[1], best[2]):
                text = f"**{text}**"
            cells.append(text)
        rows.append(f"| {n} | " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *rows])


def format_accuracy_table(acc: dict, ks=(1, 3)) -> str:
    head = "| n | " + " | ".join(acc) + " |"
    sep = "|---|" + "|".join("---" for _ in acc) + "|"
    rows = []
    for n in sorted(next(iter(acc.values()))):
        rows.append(f"| {n} | " + " | ".join(
            "{:.1%} / {:.1%}".format(*(acc[m][n][k] for k in ks)) for m in acc) + " |")
    return "\n".join([head, sep, *rows])


def accuracy_grid(counts_by_order: dict, results: dict, sentences, alpha: float = 0.4, max_positions=400) -> dict:
    """Top-1 / top-3 next-word accuracy for every method (using the tuned parameters), incl. Stupid Backoff."""
    acc = {}
    for method, by_n in results.items():
        acc[method] = {}
        for n, cell in by_n.items():
            model = CLASSES[method](counts_by_order[n], **{cell["param"][0]: cell["param"][1]})
            acc[method][n] = top_k_accuracy(model, sentences, max_positions=max_positions)
    acc["stupid-backoff"] = {n: top_k_accuracy(StupidBackoff(c, alpha), sentences, max_positions=max_positions)
                             for n, c in counts_by_order.items()}
    return acc
