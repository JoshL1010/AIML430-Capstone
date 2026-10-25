# This module implements the three AI-text detectors audited in the project.
# This code follows the published methods rather than calling a black-box tool,
# so every threshold is calibrated in the analysis instead of hidden by a vendor.
#   Fast-DetectGPT (Bao et al., 2024) measures conditional probability curvature,
#     where a HIGHER score means more machine like.
#   Binoculars (Hans et al., 2024) divides perplexity by the cross-perplexity of a base and instruct model pair, where a LOWER score means more machine-like.
#   Perplexity measures how surprising the text is to the model, the simplest baseline.
# This setup uses 0.5B-parameter Qwen2.5 models so it runs on an 8GB laptop. The
# papers use 2.7-7B models, which is discussed as a limitation in the report.

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
INSTRUCT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_TOKENS = 512
CHUNK = 64  # This limits how many positions are processed at once to save memory (mainly to save my laptop from blowing up).


# This function picks the fastest available device, trying the Apple GPU, then an NVIDIA GPU, then the CPU.
def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


# This function loads a tokenizer and language model in evaluation mode, using half precision (bfloat16) on a GPU to halve memory use.
def load_model(name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(name)
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
    return tokenizer, model


# This class stores the three detector scores for one text, plus the features the analysis uses to explain why a human text might be flagged.
@dataclass
class Scores:
    perplexity: float
    fast_detect_gpt: float
    binoculars: float
    n_tokens: int
    mean_entropy: float


# This class holds both models in memory and scores texts with all detectors.
class DetectorSuite:

    # This constructor loads the base model (used by Fast-DetectGPT, perplexity and as the Binoculars "observer") and the instruct model (the Binoculars "performer"). Both share one tokenizer, which Binoculars requires.
    def __init__(self, device: str | None = None):
        self.device = device or pick_device()
        self.tokenizer, self.base = load_model(BASE_MODEL, self.device)
        _, self.instruct = load_model(INSTRUCT_MODEL, self.device)

    # This method runs one model over a text and returns next-token logits aligned with the tokens they predict (position i predicts label i).
    @torch.no_grad()
    def _logits(self, text: str, model):
        enc = self.tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=MAX_TOKENS).to(self.device)
        return model(**enc).logits[0, :-1], enc.input_ids[0, 1:]

    # This method computes every score from one forward pass per model. The statistics are accumulated in chunks of positions, because a full probability table (512 tokens x 152k vocabulary) would not fit in memory.
    @torch.no_grad()
    def score(self, text: str) -> Scores:
        base_logits, labels = self._logits(text, self.base)
        inst_logits, _ = self._logits(text, self.instruct)

        token_ll, inst_ll, expected_ll, variance, entropy, cross_entropy = [], [], 0.0, 0.0, 0.0, 0.0
        for i in range(0, len(labels), CHUNK):
            lab = labels[i:i + CHUNK].unsqueeze(-1)
            log_p = F.log_softmax(base_logits[i:i + CHUNK].float(), dim=-1)
            p = log_p.exp()
            token_ll.append(log_p.gather(-1, lab).squeeze(-1))

            # This is the closed-form mean and variance of the log-likelihood the model expects for its own text, which Fast-DetectGPT compares against the observed log-likelihood (no sampling needed).
            e_ll = (p * log_p).sum(-1)
            expected_ll += float(e_ll.sum())
            variance += float(((p * log_p.square()).sum(-1) - e_ll.square()).clamp_min(1e-12).sum())
            entropy += float(-e_ll.sum())

            # This is the Binoculars cross-entropy, which measures how surprising the observer's predictions are to the performer over the whole vocabulary.
            inst_log_p = F.log_softmax(inst_logits[i:i + CHUNK].float(), dim=-1)
            inst_ll.append(inst_log_p.gather(-1, lab).squeeze(-1))
            cross_entropy += float(-(p * inst_log_p).sum())

        token_ll = torch.cat(token_ll)
        n = len(token_ll)
        return Scores(
            perplexity=float(torch.exp(-token_ll.mean())),
            fast_detect_gpt=float((token_ll.sum() - expected_ll) / variance ** 0.5),
            binoculars=float(-torch.cat(inst_ll).mean() / (cross_entropy / n)),
            n_tokens=n,
            mean_entropy=entropy / n,
        )


# This block is a quick self test, in which the casual human sentence should score as less machine like than the generic, formal one on all three detectors.
if __name__ == "__main__":
    human = ("I went down to the beach on Saturday because my brother had been going on "
             "about the surf all week, and honestly it was freezing, the wind came "
             "straight off the water and we lasted about twenty minutes before giving up "
             "and buying chips instead.")
    machine = ("The beach represents a significant recreational resource for coastal "
               "communities. Visitors are able to enjoy a wide range of activities, "
               "including swimming, surfing, and walking. However, it is important to "
               "consider that weather conditions can substantially affect the overall "
               "experience of any given visit.")
    suite = DetectorSuite()
    for label, text in (("human", human), ("machine", machine)):
        s = suite.score(text)
        print(f"{label:8s} perplexity={s.perplexity:6.2f}  fast-detect={s.fast_detect_gpt:6.3f}  "
              f"binoculars={s.binoculars:6.3f}")
