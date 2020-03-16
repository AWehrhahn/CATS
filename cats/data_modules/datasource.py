from os.path import expanduser
import json

from exoorbit.orbit import Orbit


class DataSource:
    def __init__(self, name=None):
        # load config file
        if name is None:
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

    def get(self, wrange, time):
        # Do the setup in __init__
        # do the calculations once its called
        raise NotImplementedError

class StellarIntensities(DataSource):
    def __init__(self, star, planet):
        super().__init__()
        self.star = star
        self.planet = planet
        self.orbit = Orbit(star, planet)
    
    def get(self, wrange, time, mode="core"):
        mu = self.orbit.phase_angle(time)
        raise NotImplementedError