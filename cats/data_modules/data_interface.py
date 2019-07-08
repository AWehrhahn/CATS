class data_interface:
    def __init__(self, configuration):
        self.configuration = configuration

class data_observations(data_interface):
    def get_observations(self, **data):
        raise NotImplementedError

class data_stellarflux(data_interface):
    def get_stellarflux(self, **data):
        raise NotImplementedError

class data_intensities(data_interface):
    def get_intensities(self, **data):
        raise NotImplementedError

class data_tellurics(data_interface):
    def get_tellurics(self, **data):
        raise NotImplementedError

class data_orbitparameters(data_interface):
    def get_parameters(self, **data):
        raise NotImplementedError

class data_orbitsimulation(data_interface):
    def get_rv(self, time, **data):
        raise NotImplementedError