"""
This script contains a function to determine which Sentinel-2 tile(s) intersect with a given area of interest (AOI) and return the corresponding S2 tile IDs.
The S2 grid is loaded from a local file or downloaded from an artifactory if it does not exist locally.
The function ensures that the AOI is in the correct projection (EPSG:4326) before performing the intersection check with the S2 grid.
"""


from pathlib import Path
import requests
import geopandas as gpd
import logging

_log = logging.getLogger(__name__)


def load_s2_grid(web_mercator: bool = False) -> gpd.GeoDataFrame:
    """Returns a geo data frame from the S2 grid."""
    # Builds the path where the geodataframe should be
    if not web_mercator:
        gdf_path = Path.home() / ".openeo-gfmap" / "s2grid_voronoi_4326.parquet"
        url = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/gfmap/s2grid_voronoi_4326.parquet"
    else:
        gdf_path = Path.home() / ".openeo-gfmap" / "s2grid_voronoi_3857.parquet"
        url = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/gfmap/s2grid_voronoi_3857.parquet"

    if not gdf_path.exists():
        _log.info("S2 grid not found, downloading it from artifactory.")
        # Downloads the file from the artifactory URL
        gdf_path.parent.mkdir(exist_ok=True)
        response = requests.get(
            url,
            timeout=180,  # 3mins
        )
        if response.status_code != 200:
            raise ValueError(
                "Failed to download the S2 grid from the artifactory. "
                f"Status code: {response.status_code}"
            )
        with open(gdf_path, "wb") as f:
            f.write(response.content)
    return gpd.read_parquet(gdf_path)


def get_s2_tile(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Check which S2 tile(s) intersect with the area of interest and return the corresponding S2 tile IDs."""

    # Ensure the gdf is in lat/lon (EPSG:4326) projection
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    # Load the S2 grid
    s2_grid = load_s2_grid()

    # Find the S2 tiles that intersect with the geometries in the gdf
    intersecting_tiles = s2_grid[
        (s2_grid.geometry.intersects(gdf.unary_union))
    ]   
    return list(intersecting_tiles["tile"].unique())


if __name__ == "__main__":

    # Example usage
    gdf = gpd.read_file("path/to/your/area_of_interest.gpkg")
    tiles = get_s2_tile(gdf)
    print(f"Intersecting S2 tiles: {tiles}")