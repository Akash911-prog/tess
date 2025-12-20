# src/utils/logging_config.py

import logging
import logging.handlers
from pathlib import Path
import sys

def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    enable_console: bool = True,
    enable_file: bool = True
):
    """
    Setup logging for the entire application.
    
    Args:
        log_level: Minimum level to log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
        enable_console: Log to console
        enable_file: Log to file
    """
    
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Get the root logger for your application
    logger = logging.getLogger("tess")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers (in case setup_logging is called multiple times)
    logger.handlers.clear()
    
    # Create formatter
    # Format: [2024-01-15 14:30:45] [INFO] [lcn] Intent matched: open chrome
    formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (colored output)
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)  # Only INFO+ to console
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler - INFO level
    if enable_file:
        info_file_handler = logging.handlers.RotatingFileHandler(
            log_path / "app.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,  # Keep 5 old files
            encoding='utf-8'
        )
        info_file_handler.setLevel(logging.INFO)
        info_file_handler.setFormatter(formatter)
        logger.addHandler(info_file_handler)
    
    # File handler - DEBUG level (more verbose)
    if enable_file and log_level.upper() == "DEBUG":
        debug_file_handler = logging.handlers.RotatingFileHandler(
            log_path / "debug.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=3,
            encoding='utf-8'
        )
        debug_file_handler.setLevel(logging.DEBUG)
        debug_file_handler.setFormatter(formatter)
        logger.addHandler(debug_file_handler)
    
    # Log the setup
    logger.info(f"Logging initialized at {log_level} level")
    
    return logger