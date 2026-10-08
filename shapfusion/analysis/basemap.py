"""Natural Earth 1:110m (v5.1.1, downloaded by ``python -m shapfusion download`` into
data/naturalearth/): land polygons for the map backgrounds, countries for the continent of
each sample (point-in-polygon; "Unknown" when a point falls in no country, e.g. small
islands or coastal points)."""
from __future__ import annotations

import numpy as np

from shapfusion.config import ROOT

NE_DIR = ROOT / "data" / "naturalearth"
_CACHE = {}


def land():
    """GeoDataFrame of the land polygons, or None (maps are then drawn without background)."""
    if "land" not in _CACHE:
        try:
            import geopandas as gpd
            _CACHE["land"] = gpd.read_file(NE_DIR / "ne_110m_land.shp")
        except Exception as e:                            # noqa: BLE001 - optional background
            print(f"  (no map background: {e})", flush=True)
            _CACHE["land"] = None
    return _CACHE["land"]


def assign_continents(lons, lats) -> np.ndarray:
    import geopandas as gpd
    world = gpd.read_file(NE_DIR / "ne_110m_admin_0_countries.shp")[["CONTINENT", "geometry"]]
    world = world[world["CONTINENT"].notna()]
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lons, lats), crs="EPSG:4326")
    joined = gpd.sjoin(pts, world, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    return joined["CONTINENT"].fillna("Unknown").values
