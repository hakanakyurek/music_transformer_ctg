import logging
import sys

def config_logging():
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        filename="logs/debug.log",
        level=logging.INFO
    )