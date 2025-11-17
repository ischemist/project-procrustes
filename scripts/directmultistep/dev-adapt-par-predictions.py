import pickle
from pathlib import Path

base_dir = Path(__file__).resolve().parents[2]
EVAL_DIR = base_dir / "data" / "evaluations"


n1_preds = EVAL_DIR / "dms-flash-fp16" / "n1" / "n1_correct_paths_NS2n.pkl"
n5_preds = EVAL_DIR / "dms-flash-fp16" / "n5" / "n5_correct_paths_NS2n.pkl"


with open(n1_preds, "rb") as f:
    n1_correct_paths: list[list[str]] = pickle.load(f)

for predictions in n1_correct_paths:
    # predictions is a list of strings representing routes predicted for a single target
    for str_route in predictions:
        route = eval(str_route)
