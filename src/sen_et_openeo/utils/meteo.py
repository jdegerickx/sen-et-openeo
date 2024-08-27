import numpy as np

from satio.timeseries import Timeseries


Z_BH = 100.0
# Acceleration of gravity (m s-2)
GRAVITY = 9.80665

# ==============================================================================
# List of constants used in Meteorological computations
# ==============================================================================
# Stephan Boltzmann constant (W m-2 K-4)
sb = 5.670373e-8
# heat capacity of dry air at constant pressure (J kg-1 K-1)
c_pd = 1003.5
# heat capacity of water vapour at constant pressure (J kg-1 K-1)
c_pv = 1865
# ratio of the molecular weight of water vapor to dry air
epsilon = 0.622
# Psicrometric Constant kPa K-1
psicr = 0.0658
# gas constant for dry air, J/(kg*degK)
R_d = 287.04
# acceleration of gravity (m s-2)
g = 9.8


def calc_c_p(p, ea):
    ''' Calculates the heat capacity of air at constant pressure.

    Parameters
    ----------
    p : float
        total air pressure (dry air + water vapour) (mb).
    ea : float
        water vapor pressure at reference height above canopy (mb).

    Returns
    -------
    c_p : heat capacity of (moist) air at constant pressure (J kg-1 K-1).

    References
    ----------
    based on equation (6.1) from Maarten Ambaum (2010):
    Thermal Physics of the Atmosphere (pp 109).'''

    # first calculate specific humidity, rearanged eq (5.22) from Maarten
    # Ambaum (2010), (pp 100)
    q = epsilon * ea / (p + (epsilon - 1.0) * ea)
    # then the heat capacity of (moist) air
    c_p = (1.0 - q) * c_pd + q * c_pv
    return np.asarray(c_p)


def calc_lambda(T_A_K):
    '''Calculates the latent heat of vaporization.

    Parameters
    ----------
    T_A_K : float
        Air temperature (Kelvin).

    Returns
    -------
    Lambda : float
        Latent heat of vaporisation (J kg-1).

    References
    ----------
    based on Eq. 3-1 Allen FAO98 '''

    Lambda = 1e6 * (2.501 - (2.361e-3 * (T_A_K - 273.15)))
    return np.asarray(Lambda)


def calc_mixing_ratio(ea, p):
    '''Calculate ratio of mass of water vapour to the mass of dry air (-)

    Parameters
    ----------
    ea : float or numpy array
        water vapor pressure at reference height (mb).
    p : float or numpy array
        total air pressure (dry air + water vapour) at reference height (mb).

    Returns
    -------
    r : float or numpy array
        mixing ratio (-)

    References
    ----------
    http://glossary.ametsoc.org/wiki/Mixing_ratio'''

    r = epsilon * ea / (p - ea)
    return r


def calc_lapse_rate_moist(T_A_K, ea, p):
    '''Calculate moist-adiabatic lapse rate (K/m)

    Parameters
    ----------
    T_A_K : float or numpy array
        air temperature at reference height (K).
    ea : float or numpy array
        water vapor pressure at reference height (mb).
    p : float or numpy array
        total air pressure (dry air + water vapour) at reference height (mb).

    Returns
    -------
    Gamma_w : float or numpy array
        moist-adiabatic lapse rate (K/m)

    References
    ----------
    http://glossary.ametsoc.org/wiki/Saturation-adiabatic_lapse_rate'''

    r = calc_mixing_ratio(ea, p)
    c_p = calc_c_p(p, ea)
    lambda_v = calc_lambda(T_A_K)
    Gamma_w = ((g * (R_d * T_A_K**2 + lambda_v * r * T_A_K)
               / (c_p * R_d * T_A_K**2 + lambda_v**2 * r * epsilon)))
    return Gamma_w


def calc_air_temperature_blending_height(ta, ea, p, z_bh, z_ta=2.0):
    if type(ta) is np.ndarray:
        ta = ta.astype(np.float32)
    if type(ea) is np.ndarray:
        ea = ea.astype(np.float32)
    if type(p) is np.ndarray:
        p = p.astype(np.float32)
    if type(z_bh) is np.ndarray:
        z_bh = z_bh.astype(np.float32)
    if type(z_ta) is np.ndarray:
        z_ta = z_ta.astype(np.float32)
    lapse_rate = calc_lapse_rate_moist(ta, ea, p)
    ta_bh = ta - lapse_rate * (z_bh - z_ta)
    return ta_bh


def comp_air_temp_inputs(ts):

    # get temp 2m
    if 't2m' not in ts.bands:
        raise ValueError('Missing input for air temperature '
                         'calculation: t2m')
    t2m = ts.select_bands(['t2m']).data

    # get geopotential height
    if 'z' not in ts.bands:
        raise ValueError('Missing input for air temperature '
                         'calculation: z')
    z = ts.select_bands(['z']).data
    z /= GRAVITY

    # get vapour pressure
    if 'vapour_pressure' not in ts.bands:
        raise ValueError('Missing input for air temperature '
                         'calculation: vapour pressure')
    ea = ts.select_bands(['vapour_pressure']).data

    # get air pressure
    if 'air_pressure' not in ts.bands:
        raise ValueError('Missing input for air temperature'
                         ' calculation: air pressure')
    p = ts.select_bands(['air_pressure']).data

    # Calcultate temperature at 0m datum height
    T_datum = calc_air_temperature_blending_height(t2m, ea, p, 0,
                                                   z_ta=z+2.0)

    # stack outputs
    outputdata = np.concatenate([ea, p, T_datum], axis=0)
    names = ['vapour_pressure', 'air_pressure', 't_datum']
    # build timeseries object
    output = Timeseries(outputdata, ts.timestamps,
                        names, ts.attrs)

    return output


def comp_air_temp(ts, elev):

    # calculate actual blending height temperature
    # based on input elevation data
    if 't_datum' not in ts.bands:
        raise ValueError('Missing input for air temperature '
                         'calculation: t_datum')
    T_datum = np.squeeze(ts.select_bands(['t_datum']).data)

    if 'air_pressure' not in ts.bands:
        raise ValueError('Missing input for air temperature'
                         ' calculation: air pressure')
    p = np.squeeze(ts.select_bands(['air_pressure']).data)

    if 'vapour_pressure' not in ts.bands:
        raise ValueError('Missing input for air temperature '
                         'calculation: vapour pressure')
    ea = np.squeeze(ts.select_bands(['vapour_pressure']).data)

    tair = calc_air_temperature_blending_height(
        T_datum, ea, p, elev+Z_BH, z_ta=0)

    # build timeseries object
    tair = np.expand_dims(tair, axis=0)
    if tair.ndim == 3:
        tair = np.expand_dims(tair, axis=0)
    output = Timeseries(tair, ts.timestamps,
                        ['air_temperature'], ts.attrs)
    return output
