"""Data engineering pipeline for the Chronically Online Translator.

This script takes the raw slang-to-standard sentence pairs, cleans the text
carefully, checks T5 token lengths, removes long outliers, and writes a clean
train/test split for downstream Hugging Face fine-tuning.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_NAME = "google/flan-t5-small"
DEFAULT_MAX_TOKENS = 64
DEFAULT_TEST_SIZE = 0.15
DEFAULT_RANDOM_STATE = 42


HIDDEN_CHAR_PATTERN = re.compile(
	"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F"
	"\u00AD\u061C\u180E\u200B-\u200F\u2028\u2029\u202A-\u202E"
	"\u2060-\u2064\u2066-\u206F\uFEFF]"
)
WHITESPACE_PATTERN = re.compile(r"\s+")
EMOJI_FALLBACK_PATTERN = re.compile(
	"(?:"
	"[\U0001F1E6-\U0001F1FF]{2}|"
	"[\U0001F300-\U0001F5FF]|"
	"[\U0001F600-\U0001F64F]|"
	"[\U0001F680-\U0001F6FF]|"
	"[\U0001F700-\U0001F77F]|"
	"[\U0001F780-\U0001F7FF]|"
	"[\U0001F800-\U0001F8FF]|"
	"[\U0001F900-\U0001F9FF]|"
	"[\U0001FA00-\U0001FAFF]|"
	"[\u2600-\u26FF]|"
	"[\u2700-\u27BF]"
	")(?:\uFE0F|\u200D[\U0001F300-\U0001FAFF]|\u200D[\u2600-\u27BF])*"
)


def configure_logging() -> None:
	"""Use a simple console logger so pipeline stats are easy to read."""

	logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
	"""Map whatever raw column names exist onto the required schema.

	The project brief assumes `source_text` and `target_text`, but the current
	raw CSV in the repository uses `normal` and `gen_z`. This function makes the
	pipeline resilient to both formats so the script can run without manual file
	edits.
	"""

	renamed = df.copy()
	lower_map = {column.lower().strip(): column for column in renamed.columns}

	aliases = {
		"source_text": ["source_text", "normal", "input", "src", "english"],
		"target_text": ["target_text", "gen_z", "slang", "output", "translation"],
	}

	resolved_columns = {}
	for canonical_name, candidate_names in aliases.items():
		matched_column = None
		for candidate_name in candidate_names:
			if candidate_name in lower_map:
				matched_column = lower_map[candidate_name]
				break
		if matched_column is None:
			if len(renamed.columns) >= 2:
				matched_column = renamed.columns[0 if canonical_name == "source_text" else 1]
			else:
				raise ValueError(
					f"Could not find a column for '{canonical_name}'. Available columns: {list(renamed.columns)}"
				)
		resolved_columns[matched_column] = canonical_name

	renamed = renamed.rename(columns=resolved_columns)
	return renamed[["source_text", "target_text"]]


def strip_hidden_characters(text: str) -> str:
	"""Remove hidden/control characters that can pollute training examples.

	We intentionally do not remove visible punctuation or emojis. The goal is to
	eliminate characters such as zero-width spaces, byte-order marks, and other
	non-printing artifacts that often sneak into CSVs and create noisy training
	pairs.
	"""

	cleaned = unicodedata.normalize("NFKC", text)
	cleaned = HIDDEN_CHAR_PATTERN.sub("", cleaned)
	return cleaned


def get_emoji_module() -> Any:
	"""Import the emoji library lazily so the script fails only at runtime if absent."""

	try:
		return importlib.import_module("emoji")
	except ModuleNotFoundError:
		return None


def get_emoji_spans(text: str) -> List[dict]:
	"""Return emoji spans using the library when present, otherwise a regex fallback."""

	emoji_module = get_emoji_module()
	if emoji_module is not None:
		return emoji_module.emoji_list(text)

	return [
		{"match_start": match.start(), "match_end": match.end()}
		for match in EMOJI_FALLBACK_PATTERN.finditer(text)
	]


def get_tokenizer(model_name: str) -> Any:
	"""Load the T5 tokenizer lazily from Hugging Face."""

	transformers_module = importlib.import_module("transformers")
	tokenizer_class = getattr(transformers_module, "T5Tokenizer")
	return tokenizer_class.from_pretrained(model_name)


def clean_text(text: object) -> str | None:
	"""Clean a single text field while preserving all emoji sequences.

	The cleaning strategy is conservative by design:
	- remove null-like values early;
	- preserve emoji spans exactly as they appear in the source text;
	- clean non-emoji spans for hidden characters and excess whitespace;
	- lowercase the final string for consistent model inputs.

	Emoji preservation matters because slang often carries meaning through emoji
	usage, and stripping them would destroy useful signal in the training data.
	"""

	if text is None or (isinstance(text, float) and pd.isna(text)):
		return None

	if not isinstance(text, str):
		text = str(text)

	text = text.strip()
	if not text:
		return None

	emoji_spans = get_emoji_spans(text)
	if not emoji_spans:
		cleaned_text = strip_hidden_characters(text)
		cleaned_text = WHITESPACE_PATTERN.sub(" ", cleaned_text).strip().lower()
		return cleaned_text or None

	preserved_segments: List[str] = []
	cursor = 0
	for span in emoji_spans:
		start = span["match_start"]
		end = span["match_end"]
		if start > cursor:
			non_emoji_segment = text[cursor:start]
			non_emoji_segment = strip_hidden_characters(non_emoji_segment)
			non_emoji_segment = WHITESPACE_PATTERN.sub(" ", non_emoji_segment)
			preserved_segments.append(non_emoji_segment)
		preserved_segments.append(text[start:end])
		cursor = end

	if cursor < len(text):
		trailing_segment = text[cursor:]
		trailing_segment = strip_hidden_characters(trailing_segment)
		trailing_segment = WHITESPACE_PATTERN.sub(" ", trailing_segment)
		preserved_segments.append(trailing_segment)

	cleaned_text = "".join(preserved_segments)
	cleaned_text = WHITESPACE_PATTERN.sub(" ", cleaned_text).strip().lower()
	return cleaned_text or None


def remove_missing_and_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
	"""Drop rows where either side of the pair is missing or empty."""

	cleaned = df.copy()
	cleaned["source_text"] = cleaned["source_text"].apply(clean_text)
	cleaned["target_text"] = cleaned["target_text"].apply(clean_text)

	cleaned = cleaned.dropna(subset=["source_text", "target_text"])
	cleaned = cleaned[
		cleaned["source_text"].astype(str).str.strip().ne("")
		& cleaned["target_text"].astype(str).str.strip().ne("")
	]
	return cleaned.reset_index(drop=True)


def get_token_lengths(texts: Sequence[str], tokenizer: Any) -> List[int]:
	"""Compute token lengths without truncation so we can inspect real sequence size."""

	encoded = tokenizer(
		list(texts),
		add_special_tokens=True,
		padding=False,
		truncation=False,
	)
	return [len(input_ids) for input_ids in encoded["input_ids"]]


def print_length_statistics(df: pd.DataFrame) -> None:
	"""Print compact statistics that can be copied into project documentation."""

	for column in ["source_token_length", "target_token_length"]:
		stats = df[column].agg(["min", "max", "mean"])
		LOGGER.info(
			"%s -> min: %d | max: %d | avg: %.2f",
			column,
			int(stats["min"]),
			int(stats["max"]),
			float(stats["mean"]),
		)


def tokenize_and_filter(
	df: pd.DataFrame,
	tokenizer: Any,
	max_tokens: int = DEFAULT_MAX_TOKENS,
) -> pd.DataFrame:
	"""Attach token lengths and remove rows that exceed the token limit.

	Keeping the sequence length bounded is important for two reasons:
	1. it prevents very long, noisy examples from dominating the dataset;
	2. it aligns the training data with a compact encoder-decoder setup.
	"""

	processed = df.copy()
	processed["source_token_length"] = get_token_lengths(processed["source_text"], tokenizer)
	processed["target_token_length"] = get_token_lengths(processed["target_text"], tokenizer)

	print_length_statistics(processed)

	before_count = len(processed)
	processed = processed[
		(processed["source_token_length"] <= max_tokens)
		& (processed["target_token_length"] <= max_tokens)
	].reset_index(drop=True)

	removed_count = before_count - len(processed)
	LOGGER.info(
		"Removed %d outlier rows longer than %d tokens in either field.",
		removed_count,
		max_tokens,
	)
	LOGGER.info("Remaining rows after token filtering: %d", len(processed))
	return processed


def split_and_save(
	df: pd.DataFrame,
	output_dir: Path,
	test_size: float = DEFAULT_TEST_SIZE,
	random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[Path, Path]:
	"""Split the dataset and write train/test CSVs for downstream fine-tuning."""

	train_df, test_df = train_test_split(
		df,
		test_size=test_size,
		random_state=random_state,
		shuffle=True,
	)

	output_dir.mkdir(parents=True, exist_ok=True)
	train_path = output_dir / "train.csv"
	test_path = output_dir / "test.csv"

	train_df[["source_text", "target_text"]].to_csv(train_path, index=False)
	test_df[["source_text", "target_text"]].to_csv(test_path, index=False)

	LOGGER.info("Saved training split to %s (%d rows).", train_path, len(train_df))
	LOGGER.info("Saved test split to %s (%d rows).", test_path, len(test_df))
	return train_path, test_path


def parse_args() -> argparse.Namespace:
	"""Expose the most useful pipeline settings as command-line flags."""

	parser = argparse.ArgumentParser(
		description="Clean slang translation pairs and prepare train/test CSVs."
	)
	parser.add_argument(
		"--input-file",
		type=Path,
		default=Path(__file__).resolve().parent / "raw" / "genz_dataset.csv",
		help="Path to the raw source/target CSV file.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path(__file__).resolve().parent / "processed",
		help="Directory where the processed train/test CSVs will be written.",
	)
	parser.add_argument(
		"--model-name",
		type=str,
		default=DEFAULT_MODEL_NAME,
		help="Hugging Face tokenizer name used to measure token lengths.",
	)
	parser.add_argument(
		"--max-tokens",
		type=int,
		default=DEFAULT_MAX_TOKENS,
		help="Maximum allowed token length for both source and target text.",
	)
	parser.add_argument(
		"--test-size",
		type=float,
		default=DEFAULT_TEST_SIZE,
		help="Fraction of rows to reserve for the test split.",
	)
	parser.add_argument(
		"--random-state",
		type=int,
		default=DEFAULT_RANDOM_STATE,
		help="Random seed for reproducible train/test splits.",
	)
	return parser.parse_args()


def main() -> None:
	"""Run the full preprocessing pipeline end to end."""

	configure_logging()
	args = parse_args()

	if not args.input_file.exists():
		raise FileNotFoundError(f"Raw dataset not found: {args.input_file}")

	LOGGER.info("Loading raw dataset from %s", args.input_file)
	raw_df = pd.read_csv(args.input_file)

	cleaned_df = standardize_columns(raw_df)
	cleaned_df = remove_missing_and_empty_rows(cleaned_df)
	LOGGER.info("Rows remaining after text cleaning: %d", len(cleaned_df))

	LOGGER.info("Loading tokenizer: %s", args.model_name)
	tokenizer = get_tokenizer(args.model_name)

	filtered_df = tokenize_and_filter(cleaned_df, tokenizer, max_tokens=args.max_tokens)

	split_and_save(
		filtered_df,
		output_dir=args.output_dir,
		test_size=args.test_size,
		random_state=args.random_state,
	)


if __name__ == "__main__":
	main()
