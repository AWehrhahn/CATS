from setuptools import setup, find_packages

import versioneer

# TODO Create and populate ~/.cats/config.json

setup(
    name="cats",
    author="Ansgar Wehrhahn",
    author_email="ansgar.wehrhahn@physics.uu.se",
    packages=find_packages(),
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
