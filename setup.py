from setuptools import setup, find_packages
from shutil import copyfile
from os.path import dirname, join, expanduser

import versioneer

# TODO Create and populate ~/.cats/config.json
config_file = expanduser("~/.cats/config.json")
local_file = join(dirname(__file__), "config.json")
copyfile(local_file, config_file)

setup(
    name="cats",
    description="Characterization of exoplanet Atmospheres with Transit Spectroscopy",
    author="Ansgar Wehrhahn",
    author_email="ansgar.wehrhahn@physics.uu.se",
    packages=find_packages(),
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
