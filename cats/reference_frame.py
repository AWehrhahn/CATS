import numpy as np
from astropy.time import Time
from astropy import units as u
from astropy import coordinates as coords

import ExoOrbit

rv_units = u.km / u.s


class ReferenceFrame:
    def from_barycentric(self):
        """
        Calculate the radial velocity from the barycentric restframe
        to this rest frame

        Returns
        -------
        rv : Quantity
            radial velocity
        """
        # This and from barycentric are just inverse to each other
        return -self.to_barycentric()

    def to_barycentric(self):
        """
        Calculate the radial velocity from this restframe to the
        barycentric restframe
        
        Retuns
        ------
        frame : BarycentricFrame
            radial velocity
        """
        raise NotImplementedError

    def to_frame(self, frame):
        return frame.from_barycentric() + self.to_barycentric()

    def from_frame(self, frame):
        return -self.to_frame(frame)


class BarycentricFrame(ReferenceFrame):
    def to_barycentric(self):
        return 0 << rv_units


class TelescopeFrame(ReferenceFrame):
    def __init__(self, datetime, observatory_location, sky_location):
        super().__init__()
        self.datetime = Time(datetime)
        self.observatory_location = observatory_location
        self.sky_location = sky_location

    @property
    def observatory(self):
        return self._observatory

    @observatory.setter
    def observatory(self, value):
        if isinstance(value, str):
            value = coords.EarthLocation.of_site(value)
        elif isinstance(value, tuple):
            lon, lat, height = value
            value = coords.EarthLocation.from_geodetic(lon, lat, height=height)

        self._observatory = value

    @property
    def sky(self):
        return self._sky

    @sky.setter
    def sky(self, value):
        if isinstance(value, tuple):
            ra, dec = value
            value = coords.SkyCoord(
                ra, dec, obstime=self.datetime, location=self.observatory
            )

        self._sky = value

    def to_barycentric(self):
        self.datetime.location = self.observatory_location
        self.sky.location = self.observatory_location
        self.sky.obstime = self.datetime

        correction = self.sky.radial_velocity_correction()
        return correction


class StarFrame(ReferenceFrame):
    def __init__(self, star):
        super().__init__()
        self.star = star

    def to_barycentric(self):
        return self.star["radial_velocity"]


class PlanetFrame(ReferenceFrame):
    def __init__(self, datetime, star, planet):
        super().__init__()
        self.datetime = Time(datetime)
        self.star = star
        self.planet = planet
        self.orbit = ExoOrbit.Orbit(star, planet)

    def to_barycentric(self):
        mjd = self.datetime.to_value("mjd", "long").value
        rv = self.orbit.radial_velocity_planet(mjd)
        return rv
