from setuptools import setup, find_packages
from shutil import copyfile
from os.path import dirname, join, expanduser
from os import makedirs
import versioneer

# TODO Create and populate ~/.cats/config.json
config_file = expanduser("~/.cats/config.json")
local_file = join(dirname(__file__), "config.json")
makedirs(dirname(config_file), exist_ok=True)
copyfile(local_file, config_file)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="cats",
    description="Characterization of exoplanet Atmospheres with Transit Spectroscopy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ansgar Wehrhahn",
    author_email="ansgar.wehrhahn@physics.uu.se",
    url="https://github.com/AWehrhahn/CATS",
    packages=find_packages(),
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Development Status :: 3 - Alpha",
    ],
)
