# This script builds the te reo Māori experiment, which compares the same essay with and without kupu Māori. Common English words are swapped for everyday te reo words (whānau, tamariki, kai, kura), and as a control the same positions are swapped for rare English words. Each essay gets four versions, the original, half swapped to te reo, fully swapped to te reo, and fully swapped to rare English.
#
# This is how the script is run
#   python src/te_reo_swap.py --input data/processed/human_sample.csv \
#                             --output data/processed/reo_variants_human.csv

import argparse
import random
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# This glossary maps each English word to (te reo Māori, rare-English control).
# Te reo nouns take no English plural "s", so plural forms map to the same kupu.
GLOSSARY = {
    "family": ("whānau", "kinfolk"), "families": ("whānau", "kinfolk"),
    "children": ("tamariki", "youngsters"), "kids": ("tamariki", "youngsters"),
    "child": ("tamaiti", "youngster"), "food": ("kai", "victuals"),
    "work": ("mahi", "labour"), "school": ("kura", "academy"),
    "schools": ("kura", "academies"), "teacher": ("kaiako", "instructor"),
    "teachers": ("kaiako", "instructors"), "student": ("ākonga", "pupil"),
    "students": ("ākonga", "pupils"), "community": ("hapori", "locality"),
    "communities": ("hapori", "localities"), "land": ("whenua", "terrain"),
    "love": ("aroha", "affection"), "conversation": ("kōrero", "discourse"),
    "conversations": ("kōrero", "discourses"), "knowledge": ("mātauranga", "erudition"),
    "learning": ("ako", "tutelage"), "group": ("rōpū", "cohort"),
    "groups": ("rōpū", "cohorts"), "elders": ("kaumātua", "seniors"),
    "ancestors": ("tūpuna", "forebears"), "home": ("kāinga", "abode"),
    "wellbeing": ("hauora", "welfare"), "environment": ("taiao", "surroundings"),
    "ocean": ("moana", "brine"), "sea": ("moana", "brine"),
    "friend": ("hoa", "companion"), "friends": ("hoa", "companions"),
}
WORD = re.compile(r"\b(" + "|".join(sorted(GLOSSARY, key=len, reverse=True)) + r")\b", re.I)


# This function replaces the glossary matches at the chosen positions with column 0 (te reo) or column 1 (rare English), keeping capital letters.
def swap(text: str, positions: set[int], column: int) -> str:
    out, last = [], 0
    for i, m in enumerate(WORD.finditer(text)):
        if i in positions:
            new = GLOSSARY[m.group(0).lower()][column]
            out += [text[last:m.start()], new[0].upper() + new[1:] if m.group(0)[0].isupper() else new]
            last = m.end()
    return "".join(out + [text[last:]])


# This function returns the four versions of one essay, each with the number of words swapped. The random half for reo_light comes from a seeded generator, so the experiment is exactly reproducible.
def variants(text: str, rng: random.Random) -> dict[str, tuple[str, int]]:
    every = set(range(len(WORD.findall(text))))
    half = set(rng.sample(sorted(every), k=(len(every) + 1) // 2)) if every else set()
    return {"original": (text, 0), "reo_light": (swap(text, half, 0), len(half)),
            "reo_full": (swap(text, every, 0), len(every)),
            "rare_english": (swap(text, every, 1), len(every))}


# This function builds the variants for every essay with at least three swappable words (fewer would barely change the text) and saves them.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=430)
    args = parser.parse_args()

    rng, rows = random.Random(args.seed), []
    for _, row in pd.read_csv(ROOT / args.input).iterrows():
        text = str(row["full_text"])
        if len(WORD.findall(text)) < 3:
            continue
        for condition, (new_text, n_swapped) in variants(text, rng).items():
            rows.append({"essay_id": row.get("essay_id", row.get("text_id")),
                         "parent_group": row.get("group"), "prompt_name": row.get("prompt_name"),
                         "condition": condition, "n_swapped": n_swapped,
                         "swap_rate": n_swapped / max(len(text.split()), 1),
                         "full_text": new_text, "group": condition})
    pd.DataFrame(rows).to_csv(ROOT / args.output, index=False)
    print(f"wrote {len(rows)} essay versions -> {args.output}")


if __name__ == "__main__":
    main()
