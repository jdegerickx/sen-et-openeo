import numpy as np


from sen_et_openeo.utils.geoloader import (getrasterinfo,
                                           writeraster,
                                           readraster)


def declination_angle(doy):
    ''' Calculates the Earth declination angle
    Parameters
    ----------
    doy : float or int
        day of the year
    Returns
    -------
    declination : float
        Declination angle (radians)
    '''
    declination = np.radians(
        23.45) * np.sin((2.0 * np.pi * doy / 365.0) - 1.39)

    return declination


def hour_angle(ftime, declination, lon, stdlon=0):
    '''Calculates the hour angle
    Parameters
    ----------
    ftime : float
        Time of the day (decimal hours)
    declination : float
        Declination angle (radians)
    lon : float
        longitude of the site (degrees).
    stdlon : float
        Longitude of the standard meridian that represent the ftime time zone
    Returns
    w : float
        hour angle (radians)
    '''

    eot = 0.258 * np.cos(declination) - 7.416 * np.sin(declination) - \
        3.648 * np.cos(2.0 * declination) - 9.228 * np.sin(2.0 * declination)
    lc = (stdlon - lon) / 15.
    time_corr = (-eot / 60.) + lc
    solar_time = ftime - time_corr
    # Get the hour angle
    w = np.radians((12.0 - solar_time) * 15.)

    return w


def incidence_angle_tilted(latfile, lonfile, doy, ftime,
                           aspectfile, slopefile,
                           stdlon=0, outfile=None):
    ''' Calculates the incidence solar angle over a tilted flat surface
    Parameters
    ----------
    lat :  float or array
        latitude (degrees)
    lon :  float or array
        longitude (degrees)
    doy : int
        day of the year
    ftime : float
        Time of the day (decimal hours)
    stdlon : float
        Longitude of the standard meridian that represent the ftime time zone
    A_ZS : float or array
        surface azimuth angle, measured clockwise from north (degrees)
    slope : float or array
        slope angle (degrees)
    Returns
    -------
    cos_theta_i : float or array
        cosine of the incidence angle
    '''

    # read raster data
    lat = readraster(latfile)
    lon = readraster(lonfile)
    aspect = readraster(aspectfile)
    slope = readraster(slopefile)
    epsg, bounds = getrasterinfo(slopefile)[0:2]

    # Get the dclination and hour angle
    delta = declination_angle(doy)
    omega = hour_angle(ftime, delta, lon, stdlon=stdlon)

    # Convert remaining angles into radians
    lat, aspect, slope = map(np.radians, [lat, aspect, slope])

    cos_theta_i = (np.sin(delta) * np.sin(lat) * np.cos(slope)
                   + np.sin(delta) * np.cos(lat) *
                   np.sin(slope) * np.cos(aspect)
                   + np.cos(delta) * np.cos(lat) *
                   np.cos(slope) * np.cos(omega)
                   - np.cos(delta) * np.sin(lat) * np.sin(slope) *
                   np.cos(aspect) * np.cos(omega)
                   - np.cos(delta) * np.sin(slope) * np.sin(aspect)
                   * np.sin(omega))

    if outfile is not None:
        # save result as rasterfile
        writeraster(cos_theta_i, outfile, epsg, bounds)
        return outfile
    else:
        return cos_theta_i
