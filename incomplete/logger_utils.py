# logger_utils.py
import logging
import os

def setup_logging(log_dir='logs'):
    """
    Sets up logging configuration with both file and console handlers.
    Creates two log files: debug.log and info.log
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture all log levels

    # If there are existing handlers, clear them to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Format for logs
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 1) File Handler (DEBUG): very detailed logs
    debug_handler = logging.FileHandler(os.path.join(log_dir, 'debug.log'))
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(formatter)
    logger.addHandler(debug_handler)

    # 2) File Handler (INFO): less verbose logs
    info_handler = logging.FileHandler(os.path.join(log_dir, 'info.log'))
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)
    logger.addHandler(info_handler)

    # 3) Console Handler (INFO): prints to stdout
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
