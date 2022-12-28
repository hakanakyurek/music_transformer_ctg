import logging
import sys

def config_logging():
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("logs/debug.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )