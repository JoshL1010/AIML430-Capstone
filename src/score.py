# This script scores every text in a CSV file with all three detectors and saves the results, keeping each text's metadata (group, prompt, essay score...) so the analysis can compare groups.
#
# This is how the script is run
#   python src/score.py --input data/processed/human_sample.csv \
#                       --output results/scores/scores_human_persuade.csv

import argparse
import time
from pathlib import Path

import pandas as pd

from detectors import DetectorSuite

ROOT = Path(__file__).resolve().parents[1]


# This function measures vocabulary diversity (the type-token ratio), which is the share of words in a text that are distinct. It is used to explain why some human writing looks machine-like.
def type_token_ratio(text: str) -> float:
    words = [w.lower().strip(".,!?;:\"'()") for w in text.split()]
    words = [w for w in words if w]
    return len(set(words)) / len(words) if words else 0.0


# This function reads the input CSV, scores each text and writes one output row per text, holding the original metadata plus the detector scores and features.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-column", default="full_text")
    args = parser.parse_args()

    df = pd.read_csv(ROOT / args.input)
    suite = DetectorSuite()
    rows, start = [], time.time()
    for i, (_, row) in enumerate(df.iterrows(), 1):
        text = str(row[args.text_column])
        s = suite.score(text)
        meta = {k: v for k, v in row.items() if k != args.text_column}
        rows.append({**meta, "word_count": len(text.split()),
                     "perplexity": s.perplexity, "fast_detect_gpt": s.fast_detect_gpt,
                     "binoculars": s.binoculars, "type_token_ratio": type_token_ratio(text),
                     "n_tokens": s.n_tokens, "mean_entropy": s.mean_entropy})
        if i % 50 == 0:
            print(f"  {i}/{len(df)}  ({(time.time() - start) / i:.2f}s per text)")

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved {len(rows)} scored texts -> {out}")


if __name__ == "__main__":
    main()
