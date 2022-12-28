import logging
import sys

def config_logging():
    logging.root.handlers = []

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler("logs/debug.log"),
            logging.StreamHandler(sys.stdout)
    ]
    )