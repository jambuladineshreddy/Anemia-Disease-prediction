import os

# Project Base Directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# File Paths
MODEL_PATH = os.path.join(BASE_DIR, "anemia_random_forest.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "anemia.csv")
BENCHMARK_DATASET_PATH = os.path.join(BASE_DIR, "anemia_raw_benchmark.csv")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")
MEDICAL_DOCS_DIR = os.path.join(BASE_DIR, "medical_docs")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")

# Feature Columns matching dataset
FEATURE_NAMES = ["Gender", "Hemoglobin", "MCH", "MCHC", "MCV"]

# Normal Reference Ranges for Lab Features
REFERENCE_RANGES = {
    "Hemoglobin": {
        "unit": "g/dL",
        "Male": (13.5, 17.5),
        "Female": (12.0, 15.5),
        "cutoff_anemia": {"Male": 13.0, "Female": 12.0}
    },
    "MCV": {
        "unit": "fL",
        "normal": (80.0, 100.0),
        "microcytic": 80.0,
        "macrocytic": 100.0
    },
    "MCH": {
        "unit": "pg",
        "normal": (27.0, 33.0)
    },
    "MCHC": {
        "unit": "g/dL",
        "normal": (32.0, 36.0)
    }
}

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # 'openai', 'ollama', or 'mock'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
