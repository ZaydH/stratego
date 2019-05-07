import logging
import sys
from pathlib import Path
from typing import Union

import torch
from torch import nn

PathOrStr = Union[Path, str]
IS_CUDA = True if torch.cuda.is_available() else False

EXPORT_DIR = Path("export")


def setup_logger(quiet_mode: bool = False, filename: PathOrStr = "test.log",
                 log_level: int = logging.DEBUG) -> None:
    r"""
    Logger Configurator

    Configures the test logger.

    :param quiet_mode: True if quiet mode (i.e., disable logging to stdout) is used
    :param filename: Log file name
    :param log_level: Level to log
    """
    if isinstance(filename, Path): filename = str(filename)

    date_format = '%m/%d/%Y %I:%M:%S %p'  # Example Time Format - 12/12/2010 11:46:36 AM
    format_str = '%(asctime)s -- %(levelname)s -- %(message)s'
    logging.basicConfig(filename=filename, level=log_level, format=format_str, datefmt=date_format)

    # Also print to stdout
    if not quiet_mode:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        formatter = logging.Formatter(format_str)
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)

    # Matplotlib clutters the logger so change its log level
    # noinspection PyProtectedMember
    # matplotlib._log.setLevel(logging.INFO)  # pylint: disable=protected-access

    logging.info("******************* New Run Beginning *****************")


def save_module(module: nn.Module, filepath: Union[Path, str]) -> None:
    r""" Save the specified \p model to disk """
    if isinstance(filepath, str):
        filepath = Path(filepath)
    filepath.parent.mkdir(exist_ok=True)
    torch.save(module.state_dict(), str(filepath))


def load_module(module: nn.Module, filepath: Union[Path, str]):
    r"""
    Loads the specified model in file \p filepath into \p module and then returns \p module.

    :param module: \p Module where the module on disk will be loaded
    :param filepath: File where the \p Module is stored
    :return: Loaded model
    """
    # Map location allows for mapping model trained on any device to be loaded
    # map_loc = lambda s, loc: 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # map_loc = lambda storage, loc: storage
    # map_loc = 'cpu'
    map_loc = 'cuda:0' if IS_CUDA else 'cpu'
    module.load_state_dict(torch.load(str(filepath), map_location=map_loc))
    # module.load_state_dict(torch.load(str(filepath)))

    module.eval()
    return module
