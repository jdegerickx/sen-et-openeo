"""
GeoJSONs layers that are used to load or visualize data.
"""
import json
import sys
if sys.version_info.minor < 7:
    from importlib_resources import open_text
else:
    from importlib.resources import open_text


_basenames = {'s2grid': 's2grid_bounds.geojson'}

layers_description = {'s2grid': 'Sentinel-2 tiles grid GeoJSON'}


class DataLayers:

    def __init__(self, layers=None, skip=[]):
        layers = layers if layers is not None else _basenames.keys()
        for k in layers:
            if k not in skip:
                exec(f'self.{k} = load("{k}")')


def load(*layers, skip=[]):
    """
    Providing only one 'layer_id' will return directly the geodataframe.

    Providing multiple layer ids (or none) will return an object where layers
    (all if none specified) can be accessed as attributes.

    Avalibale layers:

    {'s2grid': 'Sentinel-2 tiles grid GeoJSON'}
    """
    import geopandas as gpd

    if len(layers) == 1:
        with open_text('sen_et_openeo.layers', _basenames[layers[0]]) as f:
            geojson_dict = json.load(f)
            gdf = gpd.GeoDataFrame.from_features(geojson_dict, crs=4326)

            if 's2grid' in layers[0]:
                # convert column of strings to
                gdf['bounds'] = gdf['bounds'].apply(eval)

            return gdf

    elif len(layers) == 0:
        return DataLayers(None, skip=skip)
    else:
        return DataLayers(layers, skip=skip)
