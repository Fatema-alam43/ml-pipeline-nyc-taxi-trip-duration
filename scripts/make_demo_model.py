"""Train a small model on the synthetic test fixtures (no Kaggle data needed).

Used by the API tests and by CI to check that the Docker image serves predictions:
    python scripts/make_demo_model.py models/demo_model.txt
"""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import db, etl, features, train  # noqa: E402
from src.config import load_config  # noqa: E402

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "trips_sample.csv"


def train_demo_model(out_path: Path) -> Path:
    cfg = copy.deepcopy(load_config())
    cfg["model"]["improved"]["num_boost_round"] = 100
    cfg["model"]["improved"]["params"]["min_data_in_leaf"] = 10
    con = db.connect()
    clean, _ = etl.clean_sql(etl.load_raw(FIXTURE), cfg, con)
    con.close()
    model = train.train_improved(features.build_features(clean, cfg), cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_path), num_iteration=model.best_iteration)
    return out_path


if __name__ == "__main__":
    path = train_demo_model(Path(sys.argv[1] if len(sys.argv) > 1 else "models/demo_model.txt"))
    print(f"Demo model written to {path}")
