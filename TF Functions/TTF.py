import numpy as np
from scipy.interpolate import interp1d 
from acc2FAS2 import *
from kohmachi import *

def TTF(surface_acc, base_acc, dt=1e-4):
    """
    Transfer function between surface and base acceleration

    Parameters
    ----------
    surface_acc : array_like
        Surface acceleration time history
    base_acc : array_like
        Base acceleration time history
    dt : float, optional
        Time step of the acceleration time history, by default 0.01

    Returns
    -------
    freq : array_like
        Frequency vector
    TF : array_like
        Transfer function between surface and base acceleration
    """
    
    # get FAS surface
    FAS_s, freq = acc2FAS2(surface_acc, dt, 10**6)
    # downsample
    f = interp1d(freq, FAS_s)
    FAS_s = f(np.logspace(np.log10(0.1), np.log10(2.25), 256))

    # get FAS base
    FAS_b, freq = acc2FAS2(base_acc, dt, 10**6)
    # downsample
    f = interp1d(freq, FAS_b)
    FAS_b = f(np.logspace(np.log10(0.1), np.log10(2.25), 256))

    # define downsampled freq
    freq = np.logspace(np.log10(0.1), np.log10(2.25), 256)

    # get TF
    TF = kohmachi(FAS_s, freq, 150)/kohmachi(FAS_b, freq, 150)

    return freq, TF