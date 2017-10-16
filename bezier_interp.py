from scipy.interpolate import interp1d

def bezier_interp(xa, ya, y2a, x, double=None):
    """
    # Performs cubic Bezier spline interpolation
    """
    # I think thats what is supposed to happen
    return interp1d(xa, ya, kind='cubic')(x)
