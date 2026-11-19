"""Milestone 1: load, clean, tokenize and split the (entirely synthetic) administrative corpus.

Nothing here uses an external tokenizer or NLP library; it is `re` and `collections` only.
"""
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

BOS, EOS, UNK = "<s>", "</s>", "<unk>"
SPECIALS = (BOS, EOS, UNK)

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"

# Words whose trailing period belongs to the word (it never ends a sentence when a name follows).
TITLES = {"dr.", "mr.", "mrs.", "ms.", "prof.", "st.", "no.", "vs.", "tel.", "ext."}
# Other abbreviations the tokenizer keeps whole. "a.m." / "p.m." are matched by the dotted-letters rule.
ABBREVIATIONS = TITLES | {"etc.", "approx.", "dept."}

_TOKEN_RE = re.compile(
    r"""
    (?P<abbr>(?:[a-z]\.){2,})                       # a.m.  p.m.  e.g.  i.e.
  | (?P<title>(?:dr|mr|mrs|ms|prof|vs|tel|ext|approx|dept)\.)   # dr.  ms.  (no./st. are too ambiguous)
  | (?P<time>\d{1,2}:\d{2})                         # 9:30
  | (?P<num>\d+(?:[.,]\d+)*(?:st|nd|rd|th)?)        # 12  1,500  2.5  3rd
  | (?P<word>[a-z]+(?:'[a-z]+)*(?:-[a-z0-9]+(?:'[a-z]+)*)*)   # follow-up  patient's  out-of-hours
  | (?P<punct>[.,;:?!()/%&@-])
    """,
    re.VERBOSE,
)

_NO_SPACE_BEFORE = {".", ",", ";", ":", "?", "!", ")", "%"}


def clean_text(text: str) -> str:
    """Normalise whitespace and typographic characters; drop `#` comment lines."""
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    text = "\n".join(lines)
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", " - ").replace(" ", " ")
    text = re.sub(r"[\"“”]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Lowercase and split into word, number, time and punctuation tokens.

    Keeps `follow-up`, `dr.`, `a.m.`, `9:30`, `1,500`, `3rd` and `patient's` whole. A period after a known
    title/abbreviation stays attached to it; every other period becomes its own token.
    """
    return [m.group(0) for m in _TOKEN_RE.finditer(text.lower())]


def split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences. A boundary is `.`, `!` or `?` followed by a space and a capital letter,
    digit or bracket, unless the period closes a title (`Dr.`, `Ms.`) or `a.m.`/`p.m.` is followed by a title."""
    text = re.sub(r"\s+", " ", text.strip())
    sentences, start = [], 0
    for m in re.finditer(r"([.!?])\s+(?=[A-Z0-9(])", text):
        end = m.end(1)
        last = text[start:end].split()[-1].lower()
        nxt = text[m.end():].split(" ", 1)[0].lower()
        if m.group(1) == "." and (last in TITLES or (last in {"a.m.", "p.m."} and nxt in TITLES)):
            continue
        sentences.append(text[start:end].strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def detokenize(tokens: list[str]) -> str:
    """Turn tokens back into readable text: no space before `. , ; : ? ! ) %`, none after `(` or around `/`;
    hyphens stay tight. Output is lower-case, as the model sees it."""
    out = ""
    for tok in tokens:
        if tok in SPECIALS:
            continue
        if not out:
            out = tok
        elif tok in _NO_SPACE_BEFORE or tok in {"/", "-"} or out[-1] in "(/-":
            out += tok
        else:
            out += " " + tok
    return out


# --------------------------------------------------------------------------------------------------
# Corpus loading, splitting, vocabulary
# --------------------------------------------------------------------------------------------------
@dataclass
class Document:
    source: str                    # "handwritten" or "generated"
    name: str                      # file name
    sentences: list[list[str]]     # tokenized sentences (no <s>/</s>)


def load_documents(corpus_dir: Path | str = CORPUS_DIR) -> list[Document]:
    """Read every corpus/*.txt file. One non-empty line is one document (a short paragraph of 1-6 sentences);
    blank lines and `#` comment lines are ignored."""
    docs: list[Document] = []
    for path in sorted(Path(corpus_dir).glob("*.txt")):
        source = "generated" if path.stem.startswith("generated") else "handwritten"
        for line in clean_text(path.read_text(encoding="utf-8")).splitlines():
            sents = [tokenize(s) for s in split_sentences(line)]
            sents = [s for s in sents if s]
            if sents:
                docs.append(Document(source, path.name, sents))
    return docs


# Hand-written text is scarce and is what we care about, so it gets a larger held-out share than generated text.
HELD_OUT_FRACTION = {"handwritten": 0.15, "generated": 0.08}


def split_documents(docs: list[Document], seed: int = 13):
    """Deterministic split by *document* (never by sentence, so near-identical lines cannot leak across splits).

    Each source is shuffled and split separately so every split holds both handwritten and generated text.
    """
    train, val, test = [], [], []
    for source in sorted({d.source for d in docs}):
        group = [d for d in docs if d.source == source]
        random.Random(seed).shuffle(group)
        frac = HELD_OUT_FRACTION.get(source, 0.1)
        n_val = n_test = round(len(group) * frac)
        val += group[:n_val]
        test += group[n_val:n_val + n_test]
        train += group[n_val + n_test:]
    return train, val, test


def sentences_of(docs: list[Document]) -> list[list[str]]:
    return [s for d in docs for s in d.sentences]


def build_vocab(train_sentences: list[list[str]], min_count: int = 2) -> set[str]:
    """Closed vocabulary from TRAINING data only: words seen at least `min_count` times, plus the specials."""
    counts = Counter(t for s in train_sentences for t in s)
    return {w for w, c in counts.items() if c >= min_count} | set(SPECIALS)


def apply_vocab(sentences: list[list[str]], vocab: set[str]) -> list[list[str]]:
    """Replace every token outside the vocabulary with <unk> (done before counting any n-gram)."""
    return [[t if t in vocab else UNK for t in s] for s in sentences]


def corpus_stats(docs: list[Document]) -> dict:
    sents = sentences_of(docs)
    counts = Counter(t for s in sents for t in s)
    total = sum(counts.values())
    hapax = sum(1 for c in counts.values() if c == 1)
    return {
        "documents": len(docs),
        "sentences": len(sents),
        "tokens": total,
        "vocabulary": len(counts),
        "hapax_types": hapax,
        "singleton_token_share": hapax / total if total else 0.0,   # share of running tokens that occur once
        "singleton_type_share": hapax / len(counts) if counts else 0.0,
    }


@dataclass
class Prepared:
    """Everything milestone 1 hands to the model: splits as closed-vocabulary sentences plus the vocabulary."""
    vocab: set
    min_count: int
    train: list[list[str]]
    val: list[list[str]]
    test: list[list[str]]
    val_by_source: dict      # {"handwritten": [...], "generated": [...]} validation sentences (mapped to <unk>)
    test_by_source: dict
    train_by_source: dict
    docs: list[Document]
    stats: dict


def prepare(corpus_dir: Path | str = CORPUS_DIR, min_count: int = 2, seed: int = 13,
            sources: tuple = ("handwritten", "generated")) -> Prepared:
    """Load, split by document, build the vocabulary from TRAIN only, then map everything to it.

    `sources` limits which corpus parts are used (e.g. ("handwritten",) to avoid the template-generated text).
    """
    docs = [d for d in load_documents(corpus_dir) if d.source in sources]
    tr, va, te = split_documents(docs, seed=seed)
    vocab = build_vocab(sentences_of(tr), min_count)

    def by_source(ds):
        return {src: apply_vocab(sentences_of([d for d in ds if d.source == src]), vocab) for src in sources}

    return Prepared(vocab, min_count, apply_vocab(sentences_of(tr), vocab), apply_vocab(sentences_of(va), vocab),
                    apply_vocab(sentences_of(te), vocab), by_source(va), by_source(te), by_source(tr), docs,
                    corpus_stats(docs))
