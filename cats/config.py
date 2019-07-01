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


def load_config(target, filename='config.yaml'):
    """ load a configuration file
    
    The configuration file is loaded from the current working directory
    and the contents are updated with the contents of the appropiate parameter file
    
    Parameters:
    ----------
    target : {str}
        name of the star + planet combination, e.g. K2-3b
    Raises
    ------
    FileNotFoundError
        If any of the defined directories of files in config is not found
    
    Returns
    -------
    conf : dict
        configuration settings
    """
    filename = join(os.getcwd(), filename)
    conf = __load_yaml__(filename)

    if target is None:
        target = conf['name_target'] + conf['name_planet']

    data_dir = join(conf['path_exoSpectro'], target)
    conf['input_dir'] = join(data_dir, conf['dir_input'])
    conf['output_dir'] = join(data_dir, conf['dir_output'])
    par_file = join(conf['input_dir'], conf['file_parameters'])

    if not exists(data_dir):
        print('Folder for star not found', data_dir)
        raise FileNotFoundError

    if not exists(conf['input_dir']):
        print('Input directory not found', conf['input_dir'])
        raise FileNotFoundError

    if not exists(par_file):
        print('Parameter file not found, assuming default values')
    else:
        par = __load_yaml__(par_file)
        conf.update(par)

    return conf
