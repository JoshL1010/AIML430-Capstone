# Do AI text detectors treat all students fairly?

AIML430 capstone, Te Herenga Waka — Victoria University of Wellington.
Josh Lafrentz (300652648).

**Research question:** How reliable and fair are AI text detectors for students in Aotearoa, and should NZ universities use them in academic misconduct decisions?

This repository contains the code and results behind the report. Two open source detectors, implemented from their papers, are tested on human writing from before ChatGPT and on text that AI wrote, paraphrased or grammar corrected.

## Contents

```
src/detectors.py      Fast-DetectGPT, Binoculars and perplexity, implemented from the papers
src/score.py          scores a CSV of texts with all three detectors
src/te_reo_swap.py    builds the te reo Māori swap experiment (te reo vs rare English words)
notebooks/analysis.ipynb   every result and figure in the report
results/scores/       detector scores for every text (the notebook runs from these)
results/figures/      the four report figures
data/generated/frontier/   the 80 AI comparison texts (written by Claude)
```

## Running it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/analysis.ipynb   # reproduces every result from results/scores/
python src/detectors.py                     # detector self test (downloads two ~1GB models)
```

The notebook needs no GPU and runs in seconds. Rescoring texts needs the models and takes about 1.2 s per text on an M2 MacBook (the laptop im using).

## Data

All human text predates ChatGPT's release (30 Nov 2022):

| Group | Source |
|---|---|
| Native English and English learner essays (200 each), plus 1,000 held out native essays used only to set thresholds | PERSUADE 2.0 corpus (Crossley et al.), US students in grades 6 to 12, 2010 to 2020, matched on prompt and length |
| NZ English with and without te reo Māori (200 each) | .nz web pages crawled 2013 to 2020 (Common Crawl, via JoeyLLM/new-zealand-dataset-1b), same outlets, matched on length |
| Te reo Māori (200 each) | Whakaata Māori news and Salient (mC4, crawled up to 2020); Te Ara encyclopedia |

The source texts are not redistributed here. The data preparation scripts (corpus sampling, NZ and te reo collection, AI text generation) are not included for brevity and are available on request.

Two data quality problems were found and handled: the English side of the Te Ara dataset is machine translation (it is labelled as such and never counted as human writing), and most of mC4's te reo split is machine translated spam, so only Māori run news outlets were used.

## Ethics and tikanga

No human participants (confirmed with the course coordinator). Te reo Māori text is treated as Māori data under Te Mana Raraunga's principles: it is used only to test whether detectors treat te reo writers fairly, it is not used to train anything, it is not uploaded to commercial services, and no AI generated te reo was produced.