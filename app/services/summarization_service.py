"""
Abstractive text summarization using a Seq2Seq transformer model.

This module loads a HuggingFace model (tokenizer + weights) directly via
PyTorch — no `pipeline` abstraction — so every step is explicit and
inspectable: tokenization → tensor creation → generation → decoding.

Supported models (set via SUMMARIZATION_MODEL env var):
  - "distilbart"  → sshleifer/distilbart-cnn-6-6  (~300 MB, fast)
  - "flan-t5"     → google/flan-t5-base            (~900 MB, instruction-tuned)

Default: "distilbart"
"""

import os

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------------------------------------------------------------
#  Model registry
# ---------------------------------------------------------------------------
# Maps friendly names (used in env var) to HuggingFace model identifiers.
# Each entry also carries a prompt prefix: BART-based models don't need one,
# while Flan-T5 was trained with instruction prefixes — adding one improves
# output quality significantly.

MODEL_REGISTRY: dict[str, dict] = {
    "distilbart": {
        "hf_name": "sshleifer/distilbart-cnn-6-6",
        "prompt_prefix": "",  # BART expects raw text, no instruction
    },
    "flan-t5": {
        "hf_name": "google/flan-t5-base",
        "prompt_prefix": "Summarize the following text:\n\n",  # instruction prefix
    },
}

# ---------------------------------------------------------------------------
#  Resolve which model to load
# ---------------------------------------------------------------------------
_model_key = os.environ.get("SUMMARIZATION_MODEL", "distilbart").lower()

if _model_key not in MODEL_REGISTRY:
    raise ValueError(
        f"Unknown SUMMARIZATION_MODEL '{_model_key}'. "
        f"Valid options: {list(MODEL_REGISTRY.keys())}"
    )

_model_config = MODEL_REGISTRY[_model_key]
_hf_name = _model_config["hf_name"]
_prompt_prefix = _model_config["prompt_prefix"]

print(f"[summarization] Loading model: {_hf_name} ...")

# ---------------------------------------------------------------------------
#  Load tokenizer and model weights (runs once at import time)
# ---------------------------------------------------------------------------
# AutoTokenizer  — converts raw text into token IDs that the model understands.
# AutoModelForSeq2SeqLM — an encoder-decoder architecture:
#   • The ENCODER reads the input text and builds an internal representation.
#   • The DECODER generates the summary token-by-token from that representation.

tokenizer = AutoTokenizer.from_pretrained(_hf_name)
model = AutoModelForSeq2SeqLM.from_pretrained(_hf_name)

# Put the model in evaluation mode (disables dropout, batch-norm training
# behaviour etc.).  We're only doing inference, never training.
model.eval()

print(f"[summarization] Model loaded successfully: {_hf_name}")


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def summarize(text: str, max_length: int = 150, min_length: int = 40) -> str:
    """
    Generate an abstractive summary of the input text.

    Unlike extractive summarization (which picks existing sentences), this
    approach lets the model *write* a new, shorter version of the text in
    its own words.

    Args:
        text:       The original text to summarize.
        max_length: Maximum number of tokens in the generated summary.
        min_length: Minimum number of tokens — prevents overly short output.

    Returns:
        The generated summary as a plain string.

    Pipeline equivalent (for reference — we intentionally avoid this):
        >>> from transformers import pipeline
        >>> summarizer = pipeline("summarization", model=_hf_name)
        >>> summarizer(text)[0]["summary_text"]
    """

    # --- Step 1: Prepare the input text ------------------------------------
    # For instruction-tuned models (Flan-T5), prepend the task instruction.
    input_text = f"{_prompt_prefix}{text}" if _prompt_prefix else text

    # --- Step 2: Tokenize --------------------------------------------------
    # The tokenizer splits the text into sub-word tokens and returns:
    #   • input_ids      — integer IDs for each token (the model's vocabulary)
    #   • attention_mask — 1 for real tokens, 0 for padding (tells the model
    #                      which positions to attend to)
    #
    # return_tensors="pt" → returns PyTorch tensors (vs. NumPy or TF).
    # truncation=True     → cuts the text if it exceeds the model's max input
    #                       length (typically 512 or 1024 tokens).
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
    )

    # --- Step 3: Generate the summary (forward pass) -----------------------
    # torch.no_grad() tells PyTorch not to track gradients — we're not
    # training, so this saves memory and speeds things up.
    #
    # model.generate() runs the decoder in an auto-regressive loop:
    #   1. Encodes the full input via the encoder (once).
    #   2. Starts the decoder with a <BOS> (beginning-of-sequence) token.
    #   3. At each step, the decoder predicts the *next* token.
    #   4. Repeats until it produces an <EOS> token or hits max_length.
    #
    # Generation parameters explained:
    #   num_beams      — Beam search width. Instead of greedily picking the
    #                    single best token at each step (greedy search), beam
    #                    search keeps the top N candidates and explores
    #                    multiple paths in parallel. Higher = better quality
    #                    but slower.  (Try setting to 1 to see greedy output.)
    #   early_stopping — Stop as soon as all beams have produced an <EOS>,
    #                    rather than continuing until max_length.
    #   length_penalty — Values >1.0 favor longer outputs; <1.0 favor shorter.
    #                    At 2.0 the model is encouraged to write fuller
    #                    summaries instead of stopping too early.
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            early_stopping=True,
            length_penalty=2.0,
        )

    # --- Step 4: Decode the output tokens back to text ---------------------
    # output_ids is a tensor of shape (1, sequence_length).
    # We take the first (and only) sequence [0] and decode it.
    # skip_special_tokens=True removes <BOS>, <EOS>, <PAD> etc. from the
    # final string so we get clean, human-readable text.
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return summary
