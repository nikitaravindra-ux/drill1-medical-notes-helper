# Medical Notes Helper

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/z4g7ekca2v/badge.svg)](https://ask.thili.ai/b/z4g7ekca2v)

> **All text in this project is entirely synthetic.** Every sentence in `corpus/` was invented for this project.
> It contains no real patient information: no real clinical notes, no public patient dataset, nothing scraped, and
> nothing derived from a real person's record. All names, clinics, dates and places are fictional, and there are no
> identifiers (MRNs, insurance numbers, dates of birth) anywhere in the repository.
>
> **This is typing assistance for administrative wording only.** It is not a clinical decision aid and not a source
> of medical advice.

**AI assignment for the thiliAI course track project on *Language Models from scratch*: the "build a next-word
predictor from scratch" drill, by Nikita Ravindra, USN 1MS24CI079.**

## What it does

A command-line tool that helps clinic staff draft the *administrative* parts of notes and letters (appointment
scheduling, referral letters, follow-up arrangements, insurance pre-authorisation, registration wording, discharge
logistics, consent and records requests, correspondence boilerplate) by suggesting the next word. It is an n-gram
language model written from scratch with the Python standard library, with four smoothing methods compared by held-out
perplexity.

Done when (from the brief): the CLI takes user input, suggests the next word from an n-gram model trained on medical
administrative text, and applies smoothing to improve those suggestions. It does.

- [x] **1. Corpus:** synthetic administrative text, cleaned and tokenized (`app/corpus.py`, `corpus/`)
- [x] **2. N-gram model + smoothing:** add-k, Jelinek-Mercer, Stupid Backoff, interpolated Kneser-Ney (`app/ngram.py`, `app/smoothing.py`, `app/evaluate.py`)
- [x] **3. Suggestions:** ranked top-k with probabilities, prefix filter, multi-word completion (`app/suggest.py`)
- [x] **4. CLI:** `train`, `suggest`, `evaluate` and an interactive `draft` session (`app/cli.py`)

## Run it

You need Python 3.12 and git. `requirements.txt` is `httpx` and `pytest` only; the model itself uses nothing outside the
standard library.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m app.cli train          # builds models/model.json (a trained model is already committed)
python -m app.cli draft          # interactive session
pytest
```

`app/llm.py` and the `.env` / `.env.example` settings are an **unused part of the starter**. This tool calls no
external API and makes no network requests. `.env` is git-ignored and must never be committed.

## The corpus (and an honest warning about templates)

The corpus is split into two clearly labelled parts, and the numbers below are reported for each separately:

| part | how it was made | files |
|---|---|---|
| **handwritten** | Written by hand as varied prose: different sentence shapes, lengths and registers across about ten document types (phone scripts, reception notes, letters, forms, memos, SMS reminders, ...) | `corpus/handwritten_*.txt` |
| **generated** | **Template-generated**: slots filled into sentence templates by `scripts/generate_corpus.py` (seeded, deterministic) | `corpus/generated.txt` |

Slot-filled templates are far more predictable than real writing, so a model that has memorised them scores a
spectacularly low perplexity for a worthless reason. The generated part is therefore **never** used for the headline
number; it is reported separately (table B below) to show exactly that effect. Only the hand-written text is trustworthy.
Treat the generated part as extra training data that gives the model coverage of common phrases, not as evidence of quality.
The hand-written part is small (a few hundred sentences), which is the main limit on how good the suggestions can be.

Files use `#` comment headers that restate the synthetic-data statement; one non-empty line is one document. Documents
are split 80/10/10-ish into train / validation / test **by document**, never by sentence, so near-duplicate lines cannot
leak between splits (hand-written text holds out 15% for each of validation and test, generated 8%).

### Corpus statistics (whole corpus, before the split)

| part | documents | sentences | tokens | vocabulary | types seen once | singleton share of tokens |
|---|---|---|---|---|---|---|
| handwritten | 174 | 386 | 6862 | 1045 | 480 | 7.00% |
| generated | 520 | 2354 | 33755 | 457 | 13 | 0.04% |
| all | 694 | 2740 | 40617 | 1139 | 416 | 1.02% |

`singleton share of tokens` is the proportion of running tokens whose word appears only once in that part; `types seen once` counts such words. The generated part is extremely repetitive (0.04% singletons), which is the template effect in numbers.

## Tokenizer

Written by hand with `re` (`app/corpus.py`). It lowercases, then matches, in this order: dotted abbreviations
(`a.m.`, `p.m.`, `e.g.`), titles (`dr.`, `mr.`, `ms.`), times (`9:30`), numbers (`12`, `1,500`, `3rd`), words with
internal hyphens and apostrophes (`follow-up`, `out-of-hours`, `patient's`), and single punctuation marks
(`. , ; : ? ! ( ) / % & @ -`). Typographic quotes are normalised and comment lines dropped. The sentence splitter treats
`.`, `!`, `?` followed by a capital as a boundary, except after a title (`Dr. Lindqvist`) or an `a.m.`/`p.m.` that is
followed by a title. The detokenizer puts spaces back sensibly (no space before `. , ; : ? ! )`, none after `(`, hyphens
tight). Example:

```
Dr. Fenwick-Ashby's follow-up is at 9:30 a.m. on the 3rd (fee 1,500).
-> ['dr.', "fenwick-ashby's", 'follow-up', 'is', 'at', '9:30', 'a.m.', 'on', 'the', '3rd', '(', 'fee', '1,500', ')', '.']
```

## The model and the four smoothing methods

N-grams are counted over sentences padded with `n-1` copies of `<s>` and one `</s>`. `<s>` is only ever context and is
never predicted. A **closed vocabulary** is built from the *training* split only: words seen fewer than 2 times become
`<unk>`, and every word in the validation/test data that the model never saw is mapped to `<unk>` too. This happens
before any n-gram is counted, so perplexity is always finite.

| method | one-line explanation |
|---|---|
| **Add-k** (Laplace when k=1) | Pretend every possible next word was seen k extra times; no backoff at all, so unseen contexts fall back to uniform. |
| **Jelinek-Mercer** | Linear interpolation across orders: `P_n = λ·ML_n + (1-λ)·P_(n-1)`, down to a uniform distribution. |
| **Stupid Backoff** | Use the relative frequency if the n-gram was seen, otherwise `0.4 ×` the next lower order. A score, not a probability distribution. |
| **Interpolated Kneser-Ney** | Absolute discounting plus lower orders built from *continuation counts* (how many different words precede it) instead of raw frequency, so "francisco" after "san" is not wrongly predicted everywhere. |

The tests assert that the next-word distribution sums to 1.0 for add-k, Jelinek-Mercer and Kneser-Ney, and that Stupid
Backoff's does **not** (so it is excluded from the perplexity tables and compared by ranking accuracy instead).

## Evaluation: held-out perplexity, n = 1 to 5

Run `python -m app.cli evaluate` to reproduce everything below (about 15 seconds; the split and all data are
seeded). Every model within a table uses the **identical vocabulary and identical validation tokens**; `</s>` is
counted and `<s>` is not, so the token total never depends on n. Hyper-parameters are tuned per cell on the validation
set; the winner is then also scored on a held-out test set that was never used for tuning. The output is pasted
verbatim from the run:

Closed vocabulary: words seen fewer than 2 times in TRAINING data become `<unk>`; vocabulary size = 654 (incl. `<s>`, `</s>`, `<unk>`). Training sentences: 2239. Validation tokens scored: hand-written 1083, generated 3243 (`</s>` counted, `<s>` not).

### Perplexity A: trained on all text, scored on HAND-WRITTEN validation sentences (the number that matters)

Parameters in brackets were tuned on this same validation set.

| n | add-k | jelinek-mercer | kneser-ney |
|---|---|---|---|
| 1 | 193.4 (k=1) | 190.6 (lam=0.9) | 195.7 (discount=0.5) |
| 2 | 91.2 (k=0.1) | 69.8 (lam=0.6) | 87.8 (discount=0.8) |
| 3 | 173.6 (k=0.01) | 62.5 (lam=0.4) | 55.6 (discount=0.8) |
| 4 | 281.4 (k=0.01) | 62.6 (lam=0.4) | **50.4 (discount=0.8)** |
| 5 | 359.0 (k=0.01) | 63.8 (lam=0.4) | 53.4 (discount=0.9) |

### Perplexity B: same models, scored on GENERATED (template) validation sentences

Misleadingly low: the templates are memorised, so do not read this as quality.

| n | add-k | jelinek-mercer | kneser-ney |
|---|---|---|---|
| 1 | 172.5 (k=1) | 176.7 (lam=0.9) | 171.8 (discount=0.5) |
| 2 | 8.3 (k=0.1) | 8.3 (lam=0.6) | 5.4 (discount=0.8) |
| 3 | 2.8 (k=0.01) | 3.9 (lam=0.4) | 2.2 (discount=0.8) |
| 4 | 3.1 (k=0.01) | 2.7 (lam=0.4) | 2.0 (discount=0.8) |
| 5 | 3.7 (k=0.01) | 2.4 (lam=0.4) | **1.9 (discount=0.9)** |

Winner on hand-written validation: **kneser-ney, n=4**, perplexity 50.4. Held-out TEST perplexity of that setting (hand-written test sentences, never used for tuning): 54.6.

### Perplexity C: trained on HAND-WRITTEN text only (no templates anywhere), scored on hand-written validation

Vocabulary 480, training sentences 270, validation tokens 1083. Same validation sentences as A, but a DIFFERENT vocabulary (different words become `<unk>`), so these numbers must not be compared with table A; compare methods only within a table.

| n | add-k | jelinek-mercer | kneser-ney |
|---|---|---|---|
| 1 | 121.1 (k=0.001) | 123.8 (lam=0.9) | 121.1 (discount=0.7) |
| 2 | 61.8 (k=0.05) | 47.8 (lam=0.6) | 42.3 (discount=0.7) |
| 3 | 147.1 (k=0.01) | 48.5 (lam=0.4) | **39.9 (discount=0.8)** |
| 4 | 250.4 (k=0.01) | 49.6 (lam=0.4) | 40.5 (discount=0.8) |
| 5 | 312.5 (k=0.01) | 50.9 (lam=0.4) | 41.1 (discount=0.9) |

Winner: **kneser-ney, n=3**, perplexity 39.9 (test 39.7).

### Next-word accuracy A: top-1 / top-3 on hand-written validation (all methods incl. Stupid Backoff, alpha=0.4)

First 400 evenly spaced positions. Stupid Backoff has no valid perplexity (its scores do not sum to 1), so ranking accuracy is the fair way to compare it.

| n | add-k | jelinek-mercer | kneser-ney | stupid-backoff |
|---|---|---|---|---|
| 1 | 5.2% / 14.5% | 5.2% / 14.5% | 5.2% / 14.5% | 5.2% / 14.5% |
| 2 | 23.5% / 38.0% | 24.2% / 35.2% | 24.0% / 37.8% | 23.5% / 37.0% |
| 3 | 23.8% / 34.2% | 26.5% / 41.2% | 28.0% / 46.8% | 26.8% / 42.2% |
| 4 | 15.8% / 23.5% | 26.8% / 41.0% | 28.7% / 47.2% | 26.5% / 42.5% |
| 5 | 11.8% / 19.0% | 26.2% / 40.5% | 29.0% / 47.2% | 26.0% / 42.5% |

**Which combination won and why.** Interpolated Kneser-Ney with n=4 (discount 0.8) wins on hand-written validation
text (perplexity 50.4; 54.6 on the held-out test set). Kneser-Ney beats Jelinek-Mercer from n=3 upwards (at n=2 Jelinek-Mercer is actually better, 69.8 v 87.8)
because its lower-order distributions model *how many contexts a word can follow* rather than how often it occurs, which
is exactly what matters on this small corpus where most 4-grams are new. Add-k gets *worse* as n grows (193 to 359) because it spreads
mass over the whole vocabulary for every unseen context; the interpolating methods back off to shorter contexts instead.
Gains stop beyond n=4 (n=5 is slightly worse) because there is too little hand-written text for 5-grams.

Caveats, stated plainly:

- Table A and table C use **different vocabularies** and so must not be compared to each other. Within each table the comparison is fair.
- The very low perplexity in table B (about 2) is the template effect described above, not model quality.
- Perplexities of 40-60 on a ~1,000-token validation set are noisy; treat differences of a few points as ties.

## Example suggestions (real output)

Probabilities are the model's `P(word | context)` and are not renormalised after the prefix filter. An unfinished last
word is treated as a prefix, so **put a trailing space (quote the text) when the last word is finished**.

```text
$ python -m app.cli suggest "we will contact you within "
1. five                      35.5%
2. seven                     14.4%
3. three                      8.8%
4. two                        6.6%
5. ten                        6.6%

$ python -m app.cli suggest "dear dr. "
1. gallagher-moore           11.9%
2. hargreave                 11.9%
3. jessop                     5.5%
4. marchetti                  5.5%
5. valdecasas                 4.2%

$ python -m app.cli suggest "your next review is due in "
1. three                     34.5%
2. six                       30.2%
3. four                      14.5%
4. eight                     12.3%
5. two                        3.5%

$ python -m app.cli suggest "please arrive ten minutes "
1. before                    84.7%
2. ,                          2.9%
3. early                      2.8%
4. .                          2.0%
5. to                         1.3%

$ python -m app.cli suggest "authorisation has been granted for "
1. two                       38.2%
2. three                     29.7%
3. one                       26.8%
4. the                        0.9%
5. a                          0.5%

$ python -m app.cli suggest "please book a fol"
1. follow-up                 36.8%
2. folder                    <0.1%
3. following                 <0.1%

$ python -m app.cli suggest -k 3 --prefix th "please call the clinic"
1. the                        0.4%
2. that                       0.1%
3. this                       0.1%

$ python -m app.cli suggest --steps 5 -k 3 "please book a "
1. follow-up visit has been arranged         12.2%
2. copy of your records,                      1.1%
3. follow-up appointment at the front         0.6%
```

An interactive session (`python -m app.cli draft`), with input lines `your appointment`, `1`, `/multi`, `1`, `/multi`, `u`, `q`:

```text
Medical Notes Helper: typing assistance for ADMINISTRATIVE wording only. Not a clinical decision aid; not medical advice.
Type words, pick a suggestion by number, 'u' to undo, '/help' for help, 'q' to finish.

Draft: (empty)
  1. please                            15.5%
  2. we                                12.8%
  3. the                                7.9%
  4. a                                  7.6%
draft> 
Draft: your [appointment]
  1. appointment                       18.9%
  2. appointments                       0.2%
draft> 
Draft: your appointment
  1. with                              90.8%
  2. card                               3.0%
  3. has                                1.0%
  4. on                                 0.9%
draft>   multi-word completions on

Draft: your appointment
  1. with ms. quillfeather is           3.3%
  2. with ms. thornbury is              3.2%
  3. with mrs. marchetti is             3.1%
  4. with ms. quillfeather,             1.2%
draft> 
Draft: your appointment with ms. quillfeather is
  1. on the 20th of                     3.6%
  2. on friday at 3:30                  2.3%
  3. on friday at 2:15                  2.2%
  4. on the 19th of                     1.8%
draft>   multi-word completions off

Draft: your appointment with ms. quillfeather is
  1. on                                74.6%
  2. not                                1.3%
  3. available                          1.2%
  4. enclosed                           0.9%
draft>   removed: is

Draft: your appointment with ms. quillfeather
  1. is                                63.3%
  2. ,                                 22.2%
  3. at                                 3.3%
  4. will                               3.3%
draft> 
Final draft: your appointment with ms. quillfeather
```

## CLI commands

```bash
python -m app.cli hello                              # check the wiring
python -m app.cli train [--order 4] [--method kneser-ney|jelinek-mercer|add-k|stupid-backoff]
                        [--discount 0.8] [--lam 0.4] [--k 0.01] [--min-count 2]
python -m app.cli suggest "please book a fol"        # top-5 next words (prefix "fol" filters)
python -m app.cli suggest -k 3 --prefix th "please call the clinic"
python -m app.cli suggest --steps 5 -k 3 "please book a "   # multi-word completions (beam search)
python -m app.cli evaluate [--positions 400]         # full perplexity + accuracy comparison
python -m app.cli draft [-k 5]                       # interactive session
python -m app.cli --model other.json suggest ...     # use another saved model file
```

`train` fits on the whole corpus (no held-out split) and writes `models/model.json`; `suggest` and `draft` load it
instantly. Inside `draft`: type words (the last word is a filtering prefix), enter a number to accept a suggestion,
an empty line accepts what you typed, `u` undoes the last word, `/multi` toggles multi-word completions, `/new`
starts again, `/k N` changes the list length, `/help`, and `q` (or Ctrl-D) to finish. Suggestions are deterministic: ranked by
probability, ties broken alphabetically, nothing sampled.

## Project layout

```
app/
  corpus.py     milestone 1: cleaning, tokenizer, sentence splitter, detokenizer, split, closed vocabulary, stats
  ngram.py      milestone 2: n-gram counts, JSON save/load
  smoothing.py  milestone 2: AddK, JelinekMercer, StupidBackoff, KneserNey
  evaluate.py   perplexity, hyper-parameter tuning, accuracy, the report behind `evaluate`
  suggest.py    milestone 3: top-k suggestions, prefix filter, beam-search completion
  persist.py    saving / loading a whole model
  cli.py        milestone 4: train / suggest / evaluate / draft
  core.py       one function per milestone, orchestrating the modules above
  llm.py, storage.py   starter files, unused by this tool
corpus/         synthetic text (handwritten_*.txt, generated.txt)
models/         model.json, the trained model
scripts/        generate_corpus.py
tests/          test_smoke.py (starter, untouched) and test_core.py
```

## Limitations

- The hand-written corpus is small (a few hundred sentences), and the bulk of the training tokens are template-generated.
  Suggestions are best for common phrases and weakest for anything outside them; many of the "confident" suggestions
  (e.g. 85% for `before` after "please arrive ten minutes") come from the templates.
- A fixed-vocabulary n-gram model cannot invent new words and sees only the previous 3 words; it has no idea what the whole note is about.
- Names, days and numbers are ordinary words, so the model suggests whichever fictional names it saw most. Real clinics would
  train on their own wording, which this repository does not and must not contain.
- Hyper-parameters are tuned on a small validation set, so small perplexity differences are noise. Stupid Backoff has no
  valid perplexity and is compared on top-k accuracy over 400 positions only.
- Suggestions are lowercase, because the tokenizer lowercases everything.
- It is typing assistance for administrative wording. It is not a clinical tool and must not be used for medical advice.

## Settings and secrets

Model settings in the starter live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.**
This project does not need one.
