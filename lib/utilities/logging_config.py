import logging
import sys
from datetime import datetime
import os
from .constants import TASK

def config_logging(task):
    TASK = task

    logging.root.handlers = []

    os.makedirs(f'logs/{TASK}', exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=
        [
            logging.FileHandler(f"logs/{TASK}/{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}", mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )