import os
import logging

def init(fname="cats.log"):    
    """Start logging to log file and command line

    Parameters
    ----------
    log_file : str, optional
        name of the logging file (default: "log.log")
    """

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Remove existing File handles
    hasStream = False
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)
        if isinstance(h, logging.StreamHandler):
            hasStream = True

    # Command Line output
    # only if not running in notebook
    if not hasStream:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch_formatter = logging.Formatter("%(levelname)s - %(message)s")
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)

    # Log file settings
    if fname is not None:
        path = os.path.abspath(__file__)
        path = os.path.dirname(path)
        log_file = os.path.join(path, "..", "logs", fname)
        log_dir = os.path.dirname(log_file)
        if log_dir != "" and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file = logging.FileHandler(log_file)
        file.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file.setFormatter(file_formatter)
        logger.addHandler(file)

    logging.captureWarnings(True)

    logging.debug("----------------------")
