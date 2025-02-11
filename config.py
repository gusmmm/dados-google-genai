"""Configuration settings for the medical data extraction system."""

from pathlib import Path

# Model settings
MODEL_NAME = 'gemini-2.0-flash'
MODEL_TEMPERATURE = 0.0

# File paths
INPUT_DIR = Path("./dados/md_clean_files")
OUTPUT_DIR = Path("./dados/json")
DEFAULT_INPUT_FILE = INPUT_DIR / "2301.md"
GLOSSARY_PATH = Path("instructions") / "PT-glossario.md"

# Logging
LOG_FILE = "extraction.log"
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_LEVEL = "INFO"
