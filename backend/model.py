"""Local slang-to-standard translation service.

The service prefers the fine-tuned model stored in backend/fine_tuned_model/.
If the local weights are not present yet, it can fall back to a base FLAN-T5
checkpoint so the API remains testable during development.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

TASK_PREFIX = "translate slang to standard: "
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "fine_tuned_model"
DEFAULT_FALLBACK_MODEL = os.getenv("BACKEND_FALLBACK_MODEL", "google/flan-t5-small")
MAX_INPUT_LENGTH = int(os.getenv("TRANSLATION_MAX_INPUT_LENGTH", "128"))
MAX_NEW_TOKENS = int(os.getenv("TRANSLATION_MAX_NEW_TOKENS", "64"))


class ModelNotReadyError(RuntimeError):
	"""Raised when no usable translation checkpoint can be loaded."""


class TranslationService:
	"""Load a seq2seq model once and expose a simple translate method."""

	def __init__(self, model_dir: Path | str | None = None) -> None:
		self.model_dir = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
		self.tokenizer = None
		self.model = None
		self.model_source = ""
		self._load_model()

	def _weights_present(self, directory: Path) -> bool:
		"""Check for common Hugging Face weight files in a local checkpoint."""
		candidate_files = (
			"model.safetensors",
			"model.safetensors.index.json",
			"pytorch_model.bin",
			"pytorch_model.bin.index.json",
			"tf_model.h5",
			"flax_model.msgpack",
		)
		return any((directory / file_name).exists() for file_name in candidate_files)

	def _load_from_directory(self, directory: Path) -> None:
		self.tokenizer = AutoTokenizer.from_pretrained(directory)
		self.model = AutoModelForSeq2SeqLM.from_pretrained(directory)
		self.model_source = str(directory)

	def _load_model(self) -> None:
		try:
			if self.model_dir.exists() and self._weights_present(self.model_dir):
				self._load_from_directory(self.model_dir)
				return

			if os.getenv("BACKEND_ALLOW_MODEL_FALLBACK", "1") == "1":
				self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_FALLBACK_MODEL)
				self.model = AutoModelForSeq2SeqLM.from_pretrained(DEFAULT_FALLBACK_MODEL)
				self.model_source = DEFAULT_FALLBACK_MODEL
				return

			raise FileNotFoundError(
				f"No model weights found in {self.model_dir}"
			)
		except Exception as exc:  # pragma: no cover - surfaced through API startup
			raise ModelNotReadyError(
				"Unable to load a translation model. "
				"Add fine-tuned weights to backend/fine_tuned_model/ or enable fallback."
			) from exc

	def _device(self) -> torch.device:
		if self.model is None:
			return torch.device("cpu")
		return next(self.model.parameters()).device

	def translate(self, text: str) -> str:
		if self.model is None or self.tokenizer is None:
			raise ModelNotReadyError("Translation model is not available.")

		cleaned_text = text.strip()
		if not cleaned_text:
			raise ValueError("Input text cannot be empty.")

		prompt = cleaned_text
		if not cleaned_text.lower().startswith(TASK_PREFIX):
			prompt = TASK_PREFIX + cleaned_text

		inputs = self.tokenizer(
			prompt,
			return_tensors="pt",
			truncation=True,
			max_length=MAX_INPUT_LENGTH,
		)
		inputs = {key: value.to(self._device()) for key, value in inputs.items()}

		self.model.eval()
		with torch.no_grad():
			outputs = self.model.generate(
				**inputs,
				max_new_tokens=MAX_NEW_TOKENS,
				num_beams=4,
				early_stopping=True,
			)

		return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


@lru_cache(maxsize=1)
def get_translation_service() -> TranslationService:
	return TranslationService()


def translate_text(text: str) -> str:
	return get_translation_service().translate(text)
