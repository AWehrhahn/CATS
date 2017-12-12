"""
Load configuration file
"""

import os
from os.path import join

import yaml
try:
    from yaml import CLoader as Loader  # , CDumper as Dumper
except ImportError:
    print('LibYaml not installed, ')
    from yaml import Loader  # , Dumper


def __load_yaml__(fname):
    """ load json data from file with given filename """
    with open(fname, 'r') as fp:
        return yaml.load(fp, Loader=Loader)
    raise IOError

def load_config(target, filename='config.yaml'):
    """ Load configuration from file """
    filename = join(os.getcwd(), filename)
    conf = __load_yaml__(filename)

    data_dir = join(conf['path_exoSpectro'], target)
    conf['input_dir'] = join(data_dir, conf['dir_input'])
    conf['output_dir'] = join(data_dir, conf['dir_output'])
    par_file = join(conf['input_dir'], conf['file_parameters'])

    par = __load_yaml__(par_file)
    conf.update(par)
    return conf
