from pathlib import Path

def get_the_main_path() -> Path:
    return Path(__file__).resolve().parents[1]