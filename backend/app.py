"""Flask application for slang-to-standard translation."""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, request, send_from_directory
from flask_cors import CORS

from model import DEFAULT_MODEL_DIR, ModelNotReadyError, get_translation_service


def create_app() -> Flask:
	frontend_dir = (Path(__file__).resolve().parent.parent / "frontend")
	app = Flask(
		__name__,
		static_folder=str(frontend_dir),
		static_url_path="",
	)
	cors_origins = os.getenv("CORS_ORIGINS", "*")
	CORS(app, resources={r"/api/*": {"origins": cors_origins}})

	@app.get("/")
	def serve_index() -> object:
		index_file = frontend_dir / "index.html"
		if index_file.exists():
			return send_from_directory(str(frontend_dir), "index.html")
		return {"status": "ok", "message": "Frontend index.html not found."}, 404

	@app.get("/<path:asset_path>")
	def serve_assets(asset_path: str) -> object:
		asset_file = frontend_dir / asset_path
		if asset_file.exists() and asset_file.is_file():
			return send_from_directory(str(frontend_dir), asset_path)
		return {"error": "Asset not found."}, 404

	@app.get("/health")
	def health() -> tuple[dict, int]:
		return {
			"status": "ok",
			"model_directory": str(DEFAULT_MODEL_DIR),
		}, 200

	@app.post("/api/translate")
	def translate() -> tuple[dict, int]:
		payload = request.get_json(silent=True) or {}
		text = (
			payload.get("text")
			or payload.get("input")
			or payload.get("source")
			or payload.get("sentence")
		)

		if not isinstance(text, str) or not text.strip():
			return {"error": "Request body must include a non-empty text field."}, 400

		try:
			service = get_translation_service()
			translation = service.translate(text)
		except ValueError as exc:
			return {"error": str(exc)}, 400
		except ModelNotReadyError as exc:
			return {"error": str(exc)}, 503
		except Exception as exc:  # pragma: no cover - defensive API boundary
			return {"error": f"Translation failed: {exc}"}, 500

		return {
			"input": text,
			"translation": translation,
			"model_source": service.model_source,
		}, 200

	return app


app = create_app()


if __name__ == "__main__":
	host = os.getenv("HOST", "0.0.0.0")
	port = int(os.getenv("PORT", "5000"))
	debug = os.getenv("FLASK_DEBUG", "0") == "1"
	app.run(host=host, port=port, debug=debug)
