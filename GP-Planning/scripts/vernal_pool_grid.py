"""
vernal_pool_grid.py
-------------------
Minimal example: load the vernal pool shapefiles, build a grid over the site,
and keep only the points that are inside the area boundary and outside the pool
polygons (i.e., where the robot can actually go).

Usage
-----
    python scripts/vernal_pool_grid.py

The VPshapefiles/ folder should live at refs/VPshapefiles/ relative to the
project root (the same location it's used in the notebooks).

Output
------
Prints a summary and saves two files to results/:
  - traversable_grid_utm.csv   (x_utm, y_utm, lat, lon, in_traversable)
  - traversable_grid.png       quick visualisation

Dependencies: geopandas, shapely, pandas, numpy, matplotlib
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
REFS         = os.path.join(PROJECT_ROOT, "refs", "VPshapefiles")
RESULTS      = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

UTM_CRS  = "EPSG:32617"   # UTM Zone 17N  (all shapefiles reprojected to this)
LATLON   = "EPSG:4326"    # WGS-84 lat/lon

# ── Load shapefiles ────────────────────────────────────────────────────────────
# VPBoundary.shp        – outlines of the individual vernal pools
# VPBoundary_PairwiseBuffer.shp – buffered polygons used as the no-go zone
#                          (a small buffer around each pool so the robot stays
#                           clear of the pool edges)
# AreaBoundary.shp      – outer boundary of the entire sampling site
# Vernalpoolgcps*.shp   – the 31 GCP points (30 sampling + 1 VERNAL_BASE)

pool_polys = gpd.read_file(os.path.join(REFS, "VPBoundary.shp")).to_crs(UTM_CRS)
pool_bufs  = gpd.read_file(os.path.join(REFS, "VPBoundary_PairwiseBuffer.shp")).to_crs(UTM_CRS)
area_bnd   = gpd.read_file(os.path.join(REFS, "AreaBoundary.shp")).to_crs(UTM_CRS)
gcps       = gpd.read_file(
    os.path.join(REFS, "Vernalpoolgcps20260617_XYTableToPoint.shp")
).to_crs(UTM_CRS)

print(f"Pool polygons : {len(pool_polys)}")
print(f"Pool buffers  : {len(pool_bufs)}")
print(f"GCPs          : {len(gcps)}")

# ── Build traversable region ───────────────────────────────────────────────────
# traversable = area boundary  MINUS  the buffered pool no-go zones
nogo_zone       = unary_union(pool_bufs.geometry)   # union of all buffered pools
area_poly       = unary_union(area_bnd.geometry)    # outer boundary
traversable     = area_poly.difference(nogo_zone)   # the part the robot can visit

bounds = traversable.bounds  # (minx, miny, maxx, maxy) in UTM metres
print(f"\nSite bounds (UTM m): x=[{bounds[0]:.1f}, {bounds[2]:.1f}]  "
      f"y=[{bounds[1]:.1f}, {bounds[3]:.1f}]")
print(f"Site size: {bounds[2]-bounds[0]:.0f} x {bounds[3]-bounds[1]:.0f} m")

# ── Build a regular grid ───────────────────────────────────────────────────────
# Adjust GRID_SPACING to taste. At 2 m you get a fine coverage grid; at 5 m you
# get a coarser grid closer to the GCP spacing (GCPs are ~5-10 m apart).
GRID_SPACING = 2.0  # metres between grid points

xs = np.arange(bounds[0], bounds[2] + GRID_SPACING, GRID_SPACING)
ys = np.arange(bounds[1], bounds[3] + GRID_SPACING, GRID_SPACING)
XX, YY = np.meshgrid(xs, ys)
grid_utm = np.column_stack([XX.ravel(), YY.ravel()])

print(f"\nGrid: {len(xs)} x {len(ys)} = {len(grid_utm)} candidate points "
      f"at {GRID_SPACING} m spacing")

# ── Filter: keep only traversable points ──────────────────────────────────────
# A point is traversable if it's inside the area boundary AND outside all
# buffered pool polygons.  We use shapely's contains() for the exact test.
mask = np.array([traversable.contains(Point(x, y)) for x, y in grid_utm])
grid_trav = grid_utm[mask]

print(f"Traversable points: {mask.sum()} / {len(grid_utm)} "
      f"({100*mask.mean():.1f}% of bounding-box grid)")

# ── Convert UTM → lat/lon ──────────────────────────────────────────────────────
# Build a GeoDataFrame so we can use geopandas' CRS transform machinery.
gdf_all = gpd.GeoDataFrame(
    {"x_utm": grid_utm[:, 0], "y_utm": grid_utm[:, 1],
     "in_traversable": mask},
    geometry=gpd.points_from_xy(grid_utm[:, 0], grid_utm[:, 1]),
    crs=UTM_CRS,
)
gdf_latlon = gdf_all.to_crs(LATLON)
gdf_all["lat"] = gdf_latlon.geometry.y
gdf_all["lon"] = gdf_latlon.geometry.x

# ── Save CSV ───────────────────────────────────────────────────────────────────
out_csv = os.path.join(RESULTS, "traversable_grid_utm.csv")
gdf_all[["x_utm", "y_utm", "lat", "lon", "in_traversable"]].to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")

# ── Print a few traversable points as a sanity check ──────────────────────────
trav_df = gdf_all[gdf_all["in_traversable"]].reset_index(drop=True)
print(f"\nFirst 5 traversable points:")
print(trav_df[["x_utm", "y_utm", "lat", "lon"]].head().to_string(index=False))

# ── Also print the GCPs with lat/lon for reference ────────────────────────────
gcps_ll = gcps.to_crs(LATLON)
gcps["lat"] = gcps_ll.geometry.y
gcps["lon"] = gcps_ll.geometry.x
gcps["x_utm"] = gcps.geometry.x
gcps["y_utm"] = gcps.geometry.y
print(f"\nGCPs (lat/lon):")
print(gcps[["Point_ID", "x_utm", "y_utm", "lat", "lon"]].to_string(index=False))

# ── Quick visualisation ────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(8, 8))

    # Traversable region fill
    if traversable.geom_type == "MultiPolygon":
        for geom in traversable.geoms:
            xs_p, ys_p = geom.exterior.xy
            ax.fill(xs_p, ys_p, alpha=0.15, color="green")
    else:
        xs_p, ys_p = traversable.exterior.xy
        ax.fill(xs_p, ys_p, alpha=0.15, color="green")

    # Pool polygons
    pool_polys.plot(ax=ax, facecolor="#4fa3d1", edgecolor="#1a5f8a", linewidth=0.8,
                    alpha=0.7, label="Pool polygons")

    # Grid points: traversable vs excluded
    ax.scatter(grid_utm[~mask, 0], grid_utm[~mask, 1], s=1, c="#cccccc",
               alpha=0.3, label="Excluded")
    ax.scatter(trav_df["x_utm"], trav_df["y_utm"], s=2, c="#2ca02c",
               alpha=0.6, label=f"Traversable ({mask.sum()})")

    # GCP points
    ax.scatter(gcps["x_utm"], gcps["y_utm"], s=60, c="red", zorder=5,
               label="GCPs")
    for _, row in gcps.iterrows():
        ax.annotate(str(row["Point_ID"]).replace("VERNAL_", ""),
                    (row["x_utm"], row["y_utm"]),
                    fontsize=5, ha="center", va="bottom")

    ax.set_xlabel("UTM Easting (m)"); ax.set_ylabel("UTM Northing (m)")
    ax.set_title(f"Vernal Pool Site — Traversable Grid ({GRID_SPACING} m spacing)")
    ax.set_aspect("equal")
    legend_elements = [
        Patch(facecolor="#4fa3d1", label="Pool polygons"),
        Patch(facecolor="#2ca02c", alpha=0.6, label=f"Traversable ({mask.sum()} pts)"),
        Patch(facecolor="#cccccc", alpha=0.5, label="Excluded"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    out_png = os.path.join(RESULTS, "traversable_grid.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}")
    plt.close(fig)
except ImportError:
    print("matplotlib not available -- skipping visualisation")
