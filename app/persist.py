"""Save and load a trained model as JSON (counts + vocabulary + smoothing choice) so the CLI starts instantly."""
import json
from pathlib import Path

from app.ngram import NGramCounts
from app.smoothing import METHODS, LanguageModel

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.json"


def save_model(model: LanguageModel, path: Path | str = MODEL_PATH, meta: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"method": model.name, "params": model.params(), "meta": meta or {}, "counts": model.counts.to_dict()}
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return path


def load_model(path: Path | str = MODEL_PATH) -> LanguageModel:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run:  python -m app.cli train")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return METHODS[payload["method"]](NGramCounts.from_dict(payload["counts"]), **payload["params"])
