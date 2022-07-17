import logging
import sys
from typing import Optional


def get_logger_lvl(verbosity_lvl: int = 0) -> int:
    """
    Get logging lvl for logger
    :param verbosity_lvl: verbosity level
    :return: logging verbosity level
    """
    if verbosity_lvl == 1:
        level = logging.getLevelName("INFO")
    elif verbosity_lvl >= 2:
        level = logging.getLevelName("DEBUG")
    else:
        level = logging.getLevelName("WARNING")
    return level


def set_main_logger(log_file: bool = True, filename: Optional[str] = "logfile.log",
                    verbosity_lvl: Optional[int] = 0) -> None:
    """
    Set the main logger
    :param log_file: True to generate a logfile
    :param filename: logfile name (Default is "logfile.log")
    :param verbosity_lvl: level of verbosity
    """
    file_handler = logging.FileHandler(filename=filename) if log_file else None
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler] if log_file else [stdout_handler]

    level = get_logger_lvl(verbosity_lvl)

    logging.basicConfig(level=level,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=handlers)
