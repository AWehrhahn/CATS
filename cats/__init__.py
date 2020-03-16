# Load correct version string
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

# # Add output to the console
# import logging
# import colorlog
# import tqdm


# class TqdmLoggingHandler(logging.Handler):
#     def __init__(self, level=logging.NOTSET):
#         super().__init__(level)

#     def emit(self, record):
#         try:
#             msg = self.format(record)
#             tqdm.tqdm.write(msg)
#             self.flush()
#         except (KeyboardInterrupt, SystemExit):
#             raise
#         except:
#             self.handleError(record)


# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# console = TqdmLoggingHandler()
# console.setLevel(logging.INFO)
# console.setFormatter(
#     colorlog.ColoredFormatter("%(log_color)s%(levelname)s - %(message)s")
# )
# logger.addHandler(console)

# del logging
# del colorlog
