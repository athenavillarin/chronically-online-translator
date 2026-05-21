"""Evaluate the fine-tuned slang-to-standard translation model.

Metrics:
    - BLEU score (via evaluate library)
    - ROUGE-1, ROUGE-2, ROUGE-L scores (via evaluate library)
    - Cosine similarity (via sentence-transformers)

Outputs:
    - Scores printed to terminal
    - Scores saved to backend/evaluation_results.json
"""

import json
import logging
from pathlib import Path

import pandas as pd
import torch
import evaluate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
TEST_CSV        = Path("data/processed/test.csv")
MODEL_DIR       = Path("backend/fine_tuned_model")
RESULTS_FILE    = Path("backend/evaluation_results.json")

# ── Constants ──────────────────────────────────────────────────────────────────
TASK_PREFIX     = "translate slang to standard: "
MAX_INPUT_LEN   = 128
MAX_OUTPUT_LEN  = 128
SBERT_MODEL     = "all-MiniLM-L6-v2"


def load_test_data(path: Path) -> tuple[list[str], list[str]]:
    """Load source (slang) and target (standard) sentences from test CSV."""
    df = pd.read_csv(path)
    sources = df["source_text"].tolist()
    targets = df["target_text"].tolist()
    LOGGER.info(f"Loaded {len(sources)} test examples from {path}")
    return sources, targets


def generate_translations(
    sources: list[str],
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
) -> list[str]:
    """Run the fine-tuned T5 model on each source sentence."""
    translations = []
    model.eval()

    LOGGER.info("Generating translations...")
    with torch.no_grad():
        for source in sources:
            input_text = TASK_PREFIX + source
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                max_length=MAX_INPUT_LEN,
                truncation=True,
            )
            outputs = model.generate(
                inputs["input_ids"],
                max_length=MAX_OUTPUT_LEN,
                num_beams=4,
                early_stopping=True,
            )
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            translations.append(decoded)

    LOGGER.info("Translations complete.")
    return translations


def compute_bleu_rouge(
    predictions: list[str],
    references: list[str],
) -> dict:
    """Compute BLEU and ROUGE scores using the evaluate library."""
    bleu_metric  = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")

    # BLEU expects list of predictions and list of list of references
    bleu_result = bleu_metric.compute(
        predictions=predictions,
        references=[[ref] for ref in references],
    )

    rouge_result = rouge_metric.compute(
        predictions=predictions,
        references=references,
    )

    return {
        "bleu": round(bleu_result["bleu"], 4),
        "rouge1": round(rouge_result["rouge1"], 4),
        "rouge2": round(rouge_result["rouge2"], 4),
        "rougeL": round(rouge_result["rougeL"], 4),
    }


def compute_cosine_similarity(
    sources: list[str],
    translations: list[str],
) -> dict:
    """
    Compute average cosine similarity between source slang sentences
    and their translations using sentence-transformers.

    This proves semantically that the meaning was preserved post-translation.
    """
    LOGGER.info(f"Loading sentence-transformers model: {SBERT_MODEL}...")
    sbert = SentenceTransformer(SBERT_MODEL)

    source_embeddings      = sbert.encode(sources,       convert_to_tensor=True)
    translation_embeddings = sbert.encode(translations,  convert_to_tensor=True)

    cosine_scores = util.cos_sim(source_embeddings, translation_embeddings)

    # Diagonal = similarity between each source and its own translation
    per_pair_scores = [
        round(float(cosine_scores[i][i]), 4)
        for i in range(len(sources))
    ]
    avg_score = round(sum(per_pair_scores) / len(per_pair_scores), 4)

    return {
        "average_cosine_similarity": avg_score,
        "per_pair_scores": per_pair_scores,
    }


def print_results(
    sources: list[str],
    targets: list[str],
    translations: list[str],
    bleu_rouge: dict,
    cosine: dict,
) -> None:
    """Pretty-print evaluation results to terminal."""
    print("\n" + "=" * 60)
    print("  CHRONICALLY ONLINE TRANSLATOR — EVALUATION RESULTS")
    print("=" * 60)

    print("\n── Sample Translations ──────────────────────────────────")
    for i in range(min(5, len(sources))):
        print(f"\n  [{i+1}]")
        print(f"  Slang    : {sources[i]}")
        print(f"  Expected : {targets[i]}")
        print(f"  Model    : {translations[i]}")
        print(f"  Cosine   : {cosine['per_pair_scores'][i]}")

    print("\n── BLEU / ROUGE Scores ──────────────────────────────────")
    print(f"  BLEU   : {bleu_rouge['bleu']}")
    print(f"  ROUGE-1: {bleu_rouge['rouge1']}")
    print(f"  ROUGE-2: {bleu_rouge['rouge2']}")
    print(f"  ROUGE-L: {bleu_rouge['rougeL']}")

    print("\n── Semantic Similarity (Cosine) ─────────────────────────")
    print(f"  Average Cosine Similarity: {cosine['average_cosine_similarity']}")
    print("=" * 60 + "\n")


def save_results(
    bleu_rouge: dict,
    cosine: dict,
    output_path: Path,
) -> None:
    """Save all scores to a JSON file for documentation reference."""
    results = {
        "bleu_rouge": bleu_rouge,
        "cosine_similarity": cosine,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    LOGGER.info(f"Results saved to {output_path}")


def main():
    # 1. Load test data
    sources, targets = load_test_data(TEST_CSV)

    # 2. Load fine-tuned model
    LOGGER.info(f"Loading fine-tuned model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

    # 3. Generate translations
    translations = generate_translations(sources, model, tokenizer)

    # 4. Compute metrics
    LOGGER.info("Computing BLEU and ROUGE scores...")
    bleu_rouge = compute_bleu_rouge(translations, targets)

    LOGGER.info("Computing cosine similarity scores...")
    cosine = compute_cosine_similarity(sources, translations)

    # 5. Print and save
    print_results(sources, targets, translations, bleu_rouge, cosine)
    save_results(bleu_rouge, cosine, RESULTS_FILE)


if __name__ == "__main__":
    main()
