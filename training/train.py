"""Fine-tune flan-t5-small on the slang-to-standard translation task."""

import argparse
import logging
from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

MODEL_NAME = "google/flan-t5-small"
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128
TASK_PREFIX = "translate slang to standard: "


def load_data(train_path: Path, test_path: Path):
    """Load train/test CSVs into Hugging Face Dataset objects."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    LOGGER.info(f"Loaded {len(train_dataset)} training examples")
    LOGGER.info(f"Loaded {len(test_dataset)} test examples")
    
    return train_dataset, test_dataset


def preprocess_function(examples, tokenizer):
    """Tokenize inputs with task prefix and targets as labels."""
    inputs = [TASK_PREFIX + ex for ex in examples["source_text"]]
    targets = examples["target_text"]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length",
    )
    
    # Tokenize targets as labels
    labels = tokenizer(
        targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length",
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main(args):
    """Main training pipeline."""
    
    # Load model and tokenizer
    LOGGER.info(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    # Load datasets
    train_dataset, test_dataset = load_data(args.train_path, args.test_path)
    
    # Preprocess datasets
    LOGGER.info("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=["source_text", "target_text"],
    )
    test_dataset = test_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=["source_text", "target_text"],
    )
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        save_total_limit=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        predict_with_generate=True,
        fp16=False,  # Set to True if using GPU with fp16 support
        learning_rate=1e-4,
        seed=42,
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    
    # Train
    LOGGER.info("Starting training...")
    trainer.train()
    
    # Save final model
    LOGGER.info(f"Saving model to {args.output_dir}...")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    
    LOGGER.info("Training complete!")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune flan-t5-small for slang translation.")
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("data/processed/train.csv"),
        help="Path to training CSV",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path("data/processed/test.csv"),
        help="Path to test CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("backend/fine_tuned_model"),
        help="Directory to save fine-tuned model",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs (5-10 recommended)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training/evaluation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)