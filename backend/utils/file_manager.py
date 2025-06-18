import json
from pathlib import Path

DATASET_PATH = Path("datos/dataset.json")

def cargar_dataset():
    if DATASET_PATH.exists():
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"ingredientes": [], "recetas": []}

def guardar_dataset(data):
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
