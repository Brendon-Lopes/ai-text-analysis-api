"""
Abstractive text summarization using a Seq2Seq transformer model
with a Map-Reduce chunking strategy.

Why Map-Reduce?
    Models like BART and T5 have a limited context window (typically 1024
    tokens).  When the input text is longer, the tokenizer silently truncates
    it — meaning the model never even *sees* the second half of the article.
    Even within the window, BART suffers from "lead bias": it pays far more
    attention to the opening sentences because its training data (CNN/DailyMail
    news articles) follows the journalistic "inverted pyramid" structure.

    Map-Reduce solves both problems:
      1. MAP   — Split the text into small chunks and summarize each one
                  independently.  Every paragraph gets equal attention.
      2. REDUCE — Concatenate the partial summaries and run a final
                  summarization pass to produce a single, coherent output.

Supported models (set via SUMMARIZATION_MODEL env var):
  - "distilbart"  → sshleifer/distilbart-cnn-6-6  (~300 MB, fast)
  - "bart-large"  → facebook/bart-large-cnn (~1.6 GB, better quality)
  - "flan-t5"     → google/flan-t5-base            (~900 MB, instruction-tuned)

Default: "bart-large"
"""

import os

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------------------------------------------------------------
#  Model registry
# ---------------------------------------------------------------------------
# Maps friendly names (used in env var) to HuggingFace model identifiers.
# Each entry also carries:
#   - prompt_prefix: BART doesn't need one; Flan-T5 does (instruction-tuned).
#   - max_input_tokens: the encoder's context window size.

MODEL_REGISTRY: dict[str, dict] = {
    "distilbart": {
        "hf_name": "sshleifer/distilbart-cnn-6-6",
        "prompt_prefix": "",
        "max_input_tokens": 1024,
    },
    "bart-large": {
        "hf_name": "facebook/bart-large-cnn",
        "prompt_prefix": "",
        "max_input_tokens": 1024,
    },
    "flan-t5": {
        "hf_name": "google/flan-t5-base",
        "prompt_prefix": (
            "Summarize the following text in a concise and factual way. "
            "Include all main topics discussed and preserve the original "
            "meaning:\n\n"
        ),
        "max_input_tokens": 512,
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
_max_input_tokens = _model_config["max_input_tokens"]

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
#  Internal helpers
# ---------------------------------------------------------------------------

def _generate_summary(text: str, max_length: int, min_length: int, length_penalty: float = 2.0) -> str:
    """
    Run a single forward pass through the model to summarize a piece of text.

    This is the raw inference step — tokenize, generate, decode.
    Both the public `summarize()` function and the map-reduce pipeline
    call this internally.

    Args:
        text:           The text to summarize (already chunked or full).
        max_length:     Maximum number of tokens the decoder may produce.
        min_length:     Minimum number of tokens before the decoder is allowed
                        to emit an <EOS> (end-of-sequence) token.
        length_penalty: >1.0 favors longer outputs, <1.0 favors shorter outputs.

    Returns:
        The decoded summary string.
    """

    # Prepend instruction prefix for instruction-tuned models (e.g. Flan-T5).
    input_text = f"{_prompt_prefix}{text}" if _prompt_prefix else text

    # --- Tokenize ---
    # The tokenizer splits text into sub-word tokens and returns:
    #   • input_ids      — integer IDs from the model's vocabulary
    #   • attention_mask — 1 for real tokens, 0 for padding
    #
    # return_tensors="pt" → PyTorch tensors (not NumPy or TensorFlow).
    # truncation=True     → hard-cut if the chunk still exceeds max_input_tokens.
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=_max_input_tokens,
        truncation=True,
    )

    # --- Generate ---
    # torch.no_grad() disables gradient tracking (saves RAM, speeds up inference).
    #
    # model.generate() drives the decoder auto-regressively:
    #   1. Encode the full input (once).
    #   2. Start with a <BOS> token.
    #   3. Predict the next token, append it, repeat.
    #   4. Stop when <EOS> is produced or max_length is reached.
    #
    # Key parameters:
    #   num_beams          — Beam search width (4 = explore 4 paths in parallel).
    #   length_penalty     — >1.0 favors longer output; <1.0 favors shorter.
    #   no_repeat_ngram_size — Forbids repeating any 3-word sequence.
    #   early_stopping     — Stop as soon as all beams hit <EOS>.
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            length_penalty=length_penalty,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    # --- Decode ---
    # Convert token IDs back to a human-readable string.
    # skip_special_tokens=True strips <BOS>, <EOS>, <PAD>, etc.
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def _split_into_chunks(text: str) -> list[str]:
    """
    Split the input text into chunks that fit within the model's context window.

    Strategy:
      1. Split by double-newline (paragraphs) — this preserves the author's
         natural topic boundaries.
      2. If a single paragraph is too long, fall back to sentence splitting.
      3. Greedily merge small consecutive paragraphs into a single chunk
         as long as the total stays under the token limit.

    Why token-count instead of character-count?
      "Tokenization" is not 1:1 with characters or words.  The word
      "unhappiness" might be split into ["un", "happiness"] = 2 tokens.
      We must measure in tokens to guarantee the chunk fits the encoder.

    Returns:
        A list of text chunks, each guaranteed to be under the token limit.
    """

    # Reserve some tokens for the prompt prefix (if any) and safety margin.
    prefix_tokens = len(tokenizer.encode(_prompt_prefix, add_special_tokens=False))

    # IMPORTANT: We intentionally use a SMALLER chunk target than the model's
    # full context window.  The whole point of map-reduce is to force the model
    # to read each section of the text independently, giving equal attention to
    # every part.  If we used the full 1024-token window, a ~700-token article
    # would fit in a single chunk and we'd be back to the lead-bias problem.
    #
    # ~100 tokens forces the chunker to put almost every single paragraph into
    # its own chunk, preventing two topics from being merged and compressed
    # too early.
    chunk_target_tokens = 100
    max_chunk_tokens = chunk_target_tokens - prefix_tokens - 10  # 10-token safety margin

    # Step 1: Break into paragraphs (split on blank lines).
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # If the text has no paragraph breaks, fall back to splitting on sentences
    # using a simple period-based heuristic.
    if len(paragraphs) == 1:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        paragraphs = sentences

    # Step 2: Greedily merge small paragraphs into chunks that fit the window.
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_token_count = 0

    for paragraph in paragraphs:
        paragraph_tokens = len(tokenizer.encode(paragraph, add_special_tokens=False))

        # If a single paragraph exceeds the limit, it becomes its own chunk
        # (the tokenizer will truncate it during generation — better than losing
        # it entirely).
        if paragraph_tokens > max_chunk_tokens:
            # Flush whatever we've accumulated so far.
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_token_count = 0
            chunks.append(paragraph)
            continue

        # Would adding this paragraph overflow the chunk?
        if current_token_count + paragraph_tokens > max_chunk_tokens:
            # Flush the current chunk and start a new one.
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_token_count = 0

        current_chunk.append(paragraph)
        current_token_count += paragraph_tokens

    # Don't forget the last accumulated chunk.
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def summarize(text: str, max_length: int = 150, min_length: int = 40) -> str:
    """
    Generate an abstractive summary using a Map-Reduce strategy.

    For short texts (fits in one chunk), this is equivalent to a single
    model.generate() call — no overhead.

    For long texts, the process is:
      1. MAP    — Split into chunks, summarize each one independently.
      2. REDUCE — Concatenate partial summaries, summarize again to produce
                  a single coherent output.

    This eliminates the "lead bias" problem where BART only pays attention
    to the first few sentences and ignores the rest of the article.

    Args:
        text:       The original text to summarize.
        max_length: Maximum tokens in the final summary.
        min_length: Minimum tokens — prevents overly short output.

    Returns:
        The generated summary as a plain string.
    """

    chunks = _split_into_chunks(text)

    print(f"[summarization] Text split into {len(chunks)} chunk(s)")

    # --- Short text: single pass (no map-reduce needed) --------------------
    if len(chunks) == 1:
        return _generate_summary(chunks[0], max_length, min_length)

    # --- MAP phase ---------------------------------------------------------
    # The secret to preventing lead-bias in the REDUCE phase is making the MAP 
    # phase extremely terse. If we let the 8 MAP summaries be too long, merging 
    # them creates a 500-token text, and BART will simply ignore the middle 
    # (Education, Mental Health) again.
    #
    # By forcing max=50 and length_penalty=1.0, we force BART to extract precisely
    # 1 core sentence per paragraph.
    chunk_max = 50
    chunk_min = 15

    partial_summaries: list[str] = []
    for i, chunk in enumerate(chunks):
        print(f"[summarization]   MAP — summarizing chunk {i + 1}/{len(chunks)} "
              f"({len(chunk)} chars)")
        partial = _generate_summary(
            text=chunk, 
            max_length=chunk_max, 
            min_length=chunk_min, 
            length_penalty=1.0  # Encourage terse, direct extraction
        )
        partial_summaries.append(partial)

    # --- REDUCE phase ------------------------------------------------------
    # Join all partial summaries into one text and run a final summarization
    # pass.  This lets the model read ALL the key points from the entire
    # article (not just the first paragraph) and produce a coherent, unified
    # output.
    #
    # The REDUCE step gets a larger token budget than the user-requested
    # max_length to give the model room to weave all the ideas together
    # into a fluent paragraph.
    merged = " ".join(partial_summaries)

    print(f"[summarization]   REDUCE — summarizing {len(partial_summaries)} "
          f"partial summaries ({len(merged)} chars)")

    reduce_max = max(max_length, 300)
    reduce_min = max(min_length, 80)
    final_summary = _generate_summary(
        text=merged, 
        max_length=reduce_max, 
        min_length=reduce_min, 
        length_penalty=2.5  # Encourage elaborating and linking the dense ideas together
    )

    return final_summary
