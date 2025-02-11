"""Configuration settings for the medical data extraction system."""

from pathlib import Path

# Model settings
MODEL_NAME =  'gemini-2.0-flash' #'gemini-2.0-pro-exp-02-05' #
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

# Batch processing settings
MAX_FILES = None  # Set to a number to limit files processed, None for all files --> MAX_FILES = None # Process all files
MAX_WORKERS = 3   # Number of parallel workers for processing
