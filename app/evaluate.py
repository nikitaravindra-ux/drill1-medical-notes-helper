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


def report(corpus_dir=None, min_count: int = 2, seed: int = 13, accuracy_positions: int = 400) -> str:
    """Run the whole comparison and return it as Markdown (this is what `cli evaluate` prints)."""
    from app import corpus
    kwargs = {"min_count": min_count, "seed": seed}
    if corpus_dir is not None:
        kwargs["corpus_dir"] = corpus_dir
    lines = []

    def stats_line(name, st):
        return (f"| {name} | {st['documents']} | {st['sentences']} | {st['tokens']} | {st['vocabulary']} | "
                f"{st['hapax_types']} | {st['singleton_token_share']:.2%} |")

    docs = corpus.load_documents(*( [corpus_dir] if corpus_dir is not None else []))
    lines += ["### Corpus statistics (whole corpus, before the train/val/test split)", "",
              "| part | documents | sentences | tokens | vocabulary | types seen once | singleton share of tokens |",
              "|---|---|---|---|---|---|---|"]
    for src in ("handwritten", "generated"):
        part = [d for d in docs if d.source == src]
        if part:
            lines.append(stats_line(src, corpus.corpus_stats(part)))
    lines.append(stats_line("all", corpus.corpus_stats(docs)))

    # A. train on everything, tune on hand-written validation text, report both validation parts
    a = corpus.prepare(sources=("handwritten", "generated"), **kwargs)
    hw_val, gen_val = a.val_by_source["handwritten"], a.val_by_source["generated"]
    counts = build_counts(a.train, a.vocab)
    sets = {"hw val": hw_val, "gen val": gen_val, "hw test": a.test_by_source["handwritten"]}
    res_a = tune_grid(counts, hw_val, sets)
    lines += ["", f"Closed vocabulary: words seen fewer than {min_count} times in TRAINING data become `<unk>`; "
                  f"vocabulary size = {len(a.vocab)} (incl. `<s>`, `</s>`, `<unk>`). Training sentences: {len(a.train)}. "
                  f"Validation tokens scored: hand-written {token_count(hw_val)}, generated {token_count(gen_val)} "
                  f"(`</s>` counted, `<s>` not).", "",
              "### Perplexity A: trained on all text, scored on HAND-WRITTEN validation sentences (the number that matters)",
              "", "Parameters in brackets were tuned on this same validation set.", "", format_ppl_table(res_a, "hw val"),
              "", "### Perplexity B: same models, scored on GENERATED (template) validation sentences",
              "", "Misleadingly low: the templates are memorised, so do not read this as quality.", "",
              format_ppl_table(res_a, "gen val")]
    w, wm, wn = best_cell(res_a, "hw val")
    t = res_a[wm][wn]
    lines += ["", f"Winner on hand-written validation: **{wm}, n={wn}**, perplexity {w:.1f}. Held-out TEST perplexity of "
                  f"that setting (hand-written test sentences, never used for tuning): {t['ppl']['hw test']:.1f}."]

    # C. hand-written text only, so no template text is involved at all
    c = corpus.prepare(sources=("handwritten",), **kwargs)
    hw_only_val = c.val_by_source["handwritten"]
    counts_c = build_counts(c.train, c.vocab)
    res_c = tune_grid(counts_c, hw_only_val, {"hw val": hw_only_val, "hw test": c.test_by_source["handwritten"]})
    lines += ["", "### Perplexity C: trained on HAND-WRITTEN text only (no templates anywhere), scored on hand-written validation",
              "", f"Vocabulary {len(c.vocab)}, training sentences {len(c.train)}, validation tokens {token_count(hw_only_val)}. "
              "Same validation sentences as A, but a DIFFERENT vocabulary (different words become `<unk>`), so these "
              "numbers must not be compared with table A; compare methods only within a table.", "",
              format_ppl_table(res_c, "hw val")]
    w, wm, wn = best_cell(res_c, "hw val")
    lines += ["", f"Winner: **{wm}, n={wn}**, perplexity {w:.1f} (test {res_c[wm][wn]['ppl']['hw test']:.1f})."]

    # D. next-word ranking accuracy, which is also valid for Stupid Backoff
    acc = accuracy_grid(counts, res_a, hw_val, max_positions=accuracy_positions)
    lines += ["", "### Next-word accuracy A: top-1 / top-3 on hand-written validation (all methods incl. Stupid Backoff, alpha=0.4)",
              "", f"First {accuracy_positions} evenly spaced positions. Stupid Backoff has no valid perplexity (its scores do not "
              "sum to 1), so ranking accuracy is the fair way to compare it.", "", format_accuracy_table(acc)]
    return "\n".join(lines)
