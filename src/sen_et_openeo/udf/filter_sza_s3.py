from openeo.udf import XarrayDataCube
from openeo.udf.debug import inspect

MAX_SZA_DEFAULT = 90.0
SZA_BAND_DEFAULT = 'sunZenithAngles'
CODE_STR = 'filter_sza_s3_udf'


def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    """
    Filters an XarrayDataCube to remove observations with a solar zenith angle (SZA) above a specified maximum.

    Args:
        cube (XarrayDataCube): The input data cube to be processed.
        context (dict): context dictionary:
            - 'max_sza' (float): Optional. The maximum allowable solar zenith
              angle. If not provided, a default value is used.
            - 'sza_band' (str): Optional. The band to use for SZA. If not
              provided, a default value is used.

    """
    xar_in = cube.get_array()

    if 'max_sza' in context.keys():
        max_sza = context['max_sza']
    else:
        max_sza = MAX_SZA_DEFAULT

    if 'sza_band' in context.keys():
        sza_band = context['sza_band']
    else:
        sza_band = SZA_BAND_DEFAULT

    # Do filtering on SZA
    to_remove = list()
    for t in xar_in['t'].values:
        sza_data = xar_in.sel(t=t, bands=sza_band)
        max_sza_t = sza_data.max().item()
        if max_sza_t > max_sza:
            to_remove.append(t)

    inspect(
        message=f'Removing data where avg sza > {max_sza}: {to_remove}', code=CODE_STR, level='debug')
    xar_out = xar_in.drop_sel(t=to_remove)

    return XarrayDataCube(xar_out)
