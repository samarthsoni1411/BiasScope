# modules/serialization_utils.py
import pickle
from typing import Any, Dict

def save_model_wrapper(wrapper: Dict[str, Any], path: str) -> str:
    with open(path, "wb") as f:
        pickle.dump(wrapper, f)
    return path

def load_model_wrapper(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)
