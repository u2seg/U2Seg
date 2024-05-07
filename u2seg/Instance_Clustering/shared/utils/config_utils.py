# Config, logging, and other misc helper functions
import argparse
import logging
import os
from functools import partial

from config import cfg
from termcolor import cprint

logger = logging.getLogger("usl")

print_r = partial(cprint, color="red")
print_g = partial(cprint, color="green")
print_b = partial(cprint, color="blue")
print_y = partial(cprint, color="yellow")

### Logging and config parsing

# Credit: https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    blue = "\x1b[34;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: format,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def init(default_config_file):
    init_cfg(default_config_file=default_config_file)

    os.makedirs(cfg.RUN_DIR, exist_ok=cfg.SAVE_DIR_EXIST_OK)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    

# Credit: https://github.com/tqdm/tqdm/blob/master/tqdm/autonotebook.py
def is_notebook():
    import sys

    try:
        get_ipython = sys.modules['IPython'].get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            raise ImportError("console")
        if 'VSCODE_PID' in os.environ:  # pragma: no cover
            raise ImportError("vscode")
    except Exception:
        return False
    return True


# Credit: https://github.com/microsoft/Swin-Transformer/blob/main/main.py
def init_cfg(default_config_file=None):
    if is_notebook(): # Notebook mode, use default args
        cfg.merge_from_file(default_config_file)
    else:
        parser = argparse.ArgumentParser("Unsupervised Selective Labeling")
        parser.add_argument('--cfg', type=str, metavar="CFG", default=default_config_file, help='path to config')
        parser.add_argument(
            "--opts",
            help="Modify config options by adding 'KEY VALUE' pairs",
            default=None,
            nargs='+'
        )
        args = parser.parse_args()

        cfg.merge_from_file(args.cfg)
        if args.opts is not None:
            cfg.merge_from_list(args.opts)

            if cfg.OPTS_IN_RUN_NAME:
                cfg.RUN_NAME = cfg.RUN_NAME + "_" + "_".join(args.opts)

    if not cfg.RUN_DIR:
        cfg.RUN_DIR = os.path.join(cfg.SAVE_DIR, cfg.RUN_NAME)

    cfg.freeze()

