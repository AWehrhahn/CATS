"""
Load configuration file
"""

import os
from os.path import exists, join

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    print('LibYaml not installed, ')
    from yaml import Loader


def __load_yaml__(fname):
    """ load yaml data from a file

    Parameters:
    ----------
    fname : {str}
        filename of the yaml file
    Raises
    ------
    IOError
        file not found, or other io problem

    Returns
    -------
    contents: dict
        contents of the loaded yaml file
    """
    with open(fname, 'r') as fp:
        return yaml.load(fp, Loader=Loader)
    raise IOError


def load_config(star, planet, filename='config.yaml'):
    """ load a configuration file

    Parameters:
    ----------
    filename: str
        Filename of the configuration file
    **kwargs:
        keywords to use to format directory paths

    Raises
    ------
    FileNotFoundError
        If any of the defined directories of files in config is not found

    Returns
    -------
    conf : dict
        configuration settings
    """
    path = os.path.dirname(__file__)
    filename = join(path, filename)
    conf = __load_yaml__(filename)

    conf['input_dir'] = conf['path_input_data'].format(star=star, planet=planet)
    conf['output_dir'] = conf['path_output_data'].format(star=star, planet=planet)
    conf["_star"] = star
    conf["_planet"] = planet

    for k, v in conf.items():
        if isinstance(v, dict):
            conf[k]["_star"] = star
            conf[k]["_planet"] = planet

    return conf
