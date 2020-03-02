from os.path import expanduser
import json


class DataSource:
    def __init__(self):
        # load config file
        name = self.__class__.__name__.lower()
        fname = "config.json"
        fname = expanduser(f"~/.cats/{fname}")
        try:
            with open(fname, "r") as f:
                config = json.load(f)
            config = config[name]
        except (FileNotFoundError, KeyError):
            # If no configuration file exists
            # Or the key is not in the file
            config = {}

        self.config = config
