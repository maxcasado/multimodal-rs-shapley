"""
Inputs that are not in the repository, downloaded once into data/ and checked against md5:

  * the preprocessed CropHarvest binary dataset (69,800 samples; Tseng et al., NeurIPS 2021
    Datasets and Benchmarks), as distributed by the authors of DSensD+ (fmenat/DSensDp,
    data/README.md): cropharvest_binary.nc + stats/stats_cropharvest_binary.nc
  * Natural Earth 1:110m v5.1.1 (public domain): land polygons (map backgrounds) and
    countries (continent of each sample)
"""
import hashlib
import urllib.request
from pathlib import Path

DFKI = "https://cloud.dfki.de/owncloud/index.php/s/Xkd78A4BFZn9mRc/download"
NE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.1"
FILES = [
    # (path under data/, url, md5 or None)
    ("cropharvest_binary.nc", f"{DFKI}?path=%2F&files=cropharvest_binary.nc", "3e14299bd4c7f9dd5ce77e8b14b56a8a"),
    ("stats/stats_cropharvest_binary.nc", f"{DFKI}?path=%2Fstats&files=stats_cropharvest_binary.nc",
     "e164814ba5b185337b3dc2bc5d507b9a"),
    ("naturalearth/ne_110m_land.shp", f"{NE}/110m_physical/ne_110m_land.shp", "7e3f37029eea006cfa4ec3cc59fe20da"),
    ("naturalearth/ne_110m_land.shx", f"{NE}/110m_physical/ne_110m_land.shx", None),
    ("naturalearth/ne_110m_land.dbf", f"{NE}/110m_physical/ne_110m_land.dbf", None),
    ("naturalearth/ne_110m_land.prj", f"{NE}/110m_physical/ne_110m_land.prj", None),
    ("naturalearth/ne_110m_admin_0_countries.shp", f"{NE}/110m_cultural/ne_110m_admin_0_countries.shp",
     "5ae9d7eac90b47eafc13f9dea6d08315"),
    ("naturalearth/ne_110m_admin_0_countries.shx", f"{NE}/110m_cultural/ne_110m_admin_0_countries.shx",
     "e27b2572bcac814c7f416e6d23eaa7c3"),
    ("naturalearth/ne_110m_admin_0_countries.dbf", f"{NE}/110m_cultural/ne_110m_admin_0_countries.dbf",
     "14c2401f2d56e6b67b578101a1845839"),
    ("naturalearth/ne_110m_admin_0_countries.prj", f"{NE}/110m_cultural/ne_110m_admin_0_countries.prj", None),
]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(cfg):
    root = Path(cfg["data_path"])
    for rel, url, digest in FILES:
        dest = root / rel
        if dest.exists() and (digest is None or md5(dest) == digest):
            print(f"  ok        {rel}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  download  {rel}", flush=True)
        tmp = dest.with_name(dest.name + ".part")
        with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as fh:
            while chunk := r.read(1 << 20):
                fh.write(chunk)
        if digest is not None and md5(tmp) != digest:
            tmp.unlink()
            raise RuntimeError(f"{rel}: md5 mismatch for {url} (expected {digest}). If the file moved, "
                               "see 'Data' in README.md.")
        tmp.replace(dest)
    print(f"data ready in {root}")
