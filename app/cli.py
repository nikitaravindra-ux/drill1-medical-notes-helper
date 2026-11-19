"""Milestone 4: command-line interface.  `python -m app.cli --help`

    train     build the n-gram model from corpus/ and save it to models/model.json
    suggest   one-shot: print the top-k next words (or multi-word completions) for some text
    evaluate  perplexity comparison of the smoothing methods over n = 1..5
    draft     interactive drafting session with live numbered suggestions

Typing assistance for administrative wording only: not a clinical decision aid, not medical advice.
"""
import argparse
import sys
from pathlib import Path

from app import persist
from app.corpus import detokenize
from app.suggest import Suggester, split_input

BANNER = ("Medical Notes Helper: typing assistance for ADMINISTRATIVE wording only. "
          "Not a clinical decision aid; not medical advice.")

HELP = """\
How to use:
  <number>       accept that suggestion
  <text>         type words; the last word is treated as a prefix and filters the suggestions (e.g. 'fol')
  (empty line)   accept the word you typed as it is
  u  or /undo    remove the last word          /new     start a new draft
  /multi         switch between single words and multi-word completions
  /k <n>         show n suggestions            /help    this text
  q  or /quit    finish (Ctrl-D and Ctrl-C also work)
"""


def format_prob(p: float) -> str:
    return "<0.1%" if p < 0.001 else f"{p:.1%}"


class DraftSession:
    """The state of one drafting session. `handle(line)` is separate from input()/print() so it can be tested."""

    def __init__(self, model, k: int = 5, out=print):
        self.suggester = Suggester(model)
        self.k, self.out = k, out
        self.words: list[str] = []      # accepted tokens
        self.pending = ""               # a half-typed word that filters suggestions
        self.multi = False
        self.options: list[tuple] = []  # what the numbers currently mean: tuples of words

    # -- display ---------------------------------------------------------------------------------
    def text(self) -> str:
        return detokenize(self.words)

    def _context(self) -> str:
        base = self.text()
        if self.pending:
            return (base + " " if base else "") + self.pending
        return base + " " if base else ""

    def show(self) -> None:
        shown = self.text() + (" " if self.words and self.pending else "") + (f"[{self.pending}]" if self.pending else "")
        self.out(f"\nDraft: {shown or '(empty)'}")
        ctx = self._context()
        if self.multi:
            comps = self.suggester.complete(ctx, steps=4, beam=self.k)
            self.options = [c.words for c in comps]
            rows = [(detokenize(list(c.words)), c.prob) for c in comps]
        else:
            sugg = self.suggester.suggest(ctx, k=self.k)
            self.options = [(s.word,) for s in sugg]
            rows = [(s.word, s.prob) for s in sugg]
        if not rows:
            self.out("  (no suggestions for that prefix; press Enter to keep what you typed, or 'u' to undo)")
        for i, (label, p) in enumerate(rows, 1):
            self.out(f"  {i}. {label:<32}{format_prob(p):>7}")

    # -- input -----------------------------------------------------------------------------------
    def _commit_pending(self) -> None:
        if self.pending:
            self.words.append(self.pending)
            self.pending = ""

    def handle(self, line: str) -> bool:
        """Process one input line. Returns False when the session should end."""
        raw = line.rstrip("\n")
        cmd = raw.strip().lower()
        if cmd in {"q", "quit", "/q", "/quit", "exit", "/exit"}:
            return False
        if cmd in {"/help", "?", "/?", "help"}:
            self.out(HELP)
        elif cmd in {"u", "/undo", "undo"}:
            if self.pending:
                self.pending = ""
            elif self.words:
                self.out(f"  removed: {self.words.pop()}")
            else:
                self.out("  nothing to undo")
        elif cmd == "/new":
            self.words, self.pending = [], ""
            self.out("  started a new draft")
        elif cmd == "/multi":
            self.multi = not self.multi
            self.out(f"  multi-word completions {'on' if self.multi else 'off'}")
        elif cmd.startswith("/k"):
            try:
                self.k = max(1, min(20, int(cmd.split()[1])))
            except (IndexError, ValueError):
                self.out("  usage: /k <number>")
        elif cmd.isdigit():
            n = int(cmd)
            if 1 <= n <= len(self.options):
                self.pending = ""
                self.words.extend(self.options[n - 1])
            else:
                self.out(f"  choose a number between 1 and {len(self.options)}")
        elif cmd == "":
            self._commit_pending()
        else:
            self._commit_pending()
            tokens, prefix = split_input(raw if raw.endswith(" ") else raw)
            self.words.extend(tokens)
            self.pending = prefix
        return True


def run_draft(model, k: int = 5, input_fn=input, out=print) -> str:
    session = DraftSession(model, k=k, out=out)
    out(BANNER)
    out("Type words, pick a suggestion by number, 'u' to undo, '/help' for help, 'q' to finish.")
    try:
        while True:
            session.show()
            try:
                line = input_fn("draft> ")
            except (EOFError, KeyboardInterrupt):
                out("")
                break
            if not session.handle(line):
                break
    finally:
        session._commit_pending()
    final = session.text()
    out(f"\nFinal draft: {final or '(empty)'}")
    return final


# ------------------------------------------------------------------------------------------------
def _load(args):
    try:
        return persist.load_model(args.model)
    except FileNotFoundError as e:
        sys.exit(str(e))


def cmd_train(args) -> None:
    from app import core
    params = {"kneser-ney": {"discount": args.discount}, "jelinek-mercer": {"lam": args.lam},
              "add-k": {"k": args.k}, "stupid-backoff": {}}[args.method]
    model, stats = core.train_full_model(order=args.order, method=args.method, min_count=args.min_count, **params)
    path = persist.save_model(model, args.model, meta={"min_count": args.min_count, "corpus": stats})
    print(f"Trained {args.method} n={args.order} {model.params()} on {stats['sentences']} sentences, "
          f"{stats['tokens']} tokens; vocabulary {len(model.counts.vocab)} (words seen < {args.min_count}x -> <unk>).")
    try:
        path = path.relative_to(Path.cwd())
    except ValueError:
        pass
    print(f"Saved to {path}")


def cmd_suggest(args) -> None:
    from app.core import milestone_3
    model = _load(args)
    text = " ".join(args.text)
    if args.steps:
        for i, c in enumerate(milestone_3(model, text, steps=args.steps, beam=args.k), 1):
            print(f"{i}. {detokenize(list(c.words)):<40}{format_prob(c.prob):>7}")
    else:
        for i, s in enumerate(milestone_3(model, text, k=args.k, prefix=args.prefix), 1):
            print(f"{i}. {s.word:<24}{format_prob(s.prob):>7}")


def cmd_evaluate(args) -> None:
    from app import evaluate
    print(evaluate.report(accuracy_positions=args.positions))


def cmd_draft(args) -> None:
    run_draft(_load(args), k=args.k)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="medical-notes-helper", description=(
        "Next-word suggestions for drafting ADMINISTRATIVE sections of notes (n-gram model with smoothing). "
        "Typing assistance only: not a clinical decision aid and not medical advice."))
    parser.add_argument("--model", default=str(persist.MODEL_PATH), help="path of the saved model JSON")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")

    p = sub.add_parser("train", help="train on corpus/ and save the model")
    p.add_argument("--order", type=int, default=4)
    p.add_argument("--method", choices=["kneser-ney", "jelinek-mercer", "add-k", "stupid-backoff"], default="kneser-ney")
    p.add_argument("--discount", type=float, default=0.8, help="Kneser-Ney discount")
    p.add_argument("--lam", type=float, default=0.4, help="Jelinek-Mercer weight")
    p.add_argument("--k", type=float, default=0.01, help="add-k constant")
    p.add_argument("--min-count", type=int, default=2, help="words seen fewer times become <unk>")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("suggest", help="print the top-k next words for the given text")
    p.add_argument("text", nargs="*", help="what has been typed so far (an unfinished last word acts as a prefix)")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--prefix", help="only suggest words starting with this")
    p.add_argument("--steps", type=int, default=0, help="multi-word mode: extend this many words ahead")
    p.set_defaults(func=cmd_suggest)

    p = sub.add_parser("evaluate", help="perplexity comparison of the smoothing methods")
    p.add_argument("--positions", type=int, default=400, help="positions used for the accuracy table")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("draft", help="interactive drafting session")
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(func=cmd_draft)

    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Medical Notes Helper is ready. Open README.md for your milestones.")
    else:
        args.func(args)


if __name__ == "__main__":
    main()
