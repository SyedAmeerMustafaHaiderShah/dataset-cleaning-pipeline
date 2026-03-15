"""
logger.py
---------
Handles all logging for the data cleaning pipeline.
Every operation, row count change, and step is recorded
in a timestamped log file saved to the output folder.
"""

import logging
import os
from datetime import datetime


def setup_logger(output_dir: str) -> logging.Logger:
    """
    Creates and configures a logger that writes to both
    the console and a timestamped log file in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(output_dir, f"pipeline_log_{timestamp}.txt")

    logger = logging.getLogger("DataCleaningPipeline")
    logger.setLevel(logging.DEBUG)

    # Clear any existing handlers (important if re-running in same session)
    logger.handlers.clear()

    # --- File Handler (saves everything) ---
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    # --- Console Handler (shows INFO and above) ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"📄 Log file created: {log_filename}")
    return logger, log_filename
