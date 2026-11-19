# Medical Notes Helper

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/z4g7ekca2v/badge.svg)](https://ask.thili.ai/b/z4g7ekca2v)

The Medical Notes Helper is a CLI tool designed for clinic staff in the Healthcare domain. It assists in drafting administrative medical notes by providing next-word suggestions, aiming to improve consistency and reduce manual typing effort.

## What you are building

**Goal.** You will build a Command Line Interface (CLI) tool that helps clinic staff draft administrative sections of medical notes by generating next-word suggestions.

**Done when.** The project is done when you have a CLI tool that can take user input, suggest the next word based on an n-gram language model trained on medical text, and apply smoothing for better suggestions.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Curate an anonymized corpus of administrative medical text.**  
      Hint: Corpus preparation: Focus on cleaning and tokenizing your text data.
- [ ] **2. Implement the n-gram language model and apply smoothing.**  
      Hint: N-gram model: Implement the n-gram probabilities and apply smoothing techniques.
- [ ] **3. Develop a text generation feature that suggests words based on input.**  
      Hint: Text generation: Use your n-gram model to predict the next word given a sequence.
- [ ] **4. Create a CLI for users to interact with the suggestion engine.**  
      Hint: CLI development: Create an interactive command-line interface for user input and suggestions.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- (none)
