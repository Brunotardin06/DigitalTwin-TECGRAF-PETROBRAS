"""
GIFs por par (forecast x hindcast) desenhando um POLIGONO preenchido que
envolve as particulas de cada mancha -> aparencia de mancha de oleo.

O poligono e a uniao de buffers ao redor de cada ponto ativo (garante que
todo ponto fica DENTRO do poligono). Extensao do mapa fixa contendo tudo +
margem (inclui o raio do buffer) -> nada e cortado.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
import numpy as np
import xarray as xr
import imageio.v2 as imageio
from shapely import concave_hull
from shapely.geometry import MultiPoint

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "2-simulated" / "forecast_hindcast_pairs"
GIF_DIR = D / "gifs_polygon"
GIF_DIR.mkdir(parents=True, exist_ok=True)
FRAME_TMP = GIF_DIR / "_frames"

FC_COLOR = "#1f77b4"   # forecast = azul
HC_COLOR = "#d62728"   # hindcast = vermelho
FPS = 4
MARGIN_FRAC = 0.08


def find_pairs():
    pairs = {}
    for f in sorted(D.glob("pair*_forecast_*.nc")):
        m = re.match(r"(pair\d+)_(\d{4}_\d{2})_forecast_(\d+T\d+)_(\d+T\d+)\.nc", f.name)
        if not m:
            continue
        key, tag, _, _ = m.groups()
        hc = f.with_name(f.name.replace("_forecast_", "_hindcast_"))
        if hc.exists():
            pairs[key] = (tag, f, hc)
    return dict(sorted(pairs.items()))


def load_xy(path):
    ds = xr.open_dataset(path)
    lon = ds["lon"].values.astype(float)
    lat = ds["lat"].values.astype(float)
    times = ds["time"].values
    ds.close()
    return lon, lat, times


def buffer_radius(fc, hc):
    """Raio (graus) do buffer, proporcional a dispersao total da mancha."""
    lon = np.concatenate([fc[0].ravel(), hc[0].ravel()])
    lat = np.concatenate([fc[1].ravel(), hc[1].ravel()])
    dlon = np.nanmax(lon) - np.nanmin(lon)
    dlat = np.nanmax(lat) - np.nanmin(lat)
    r = 0.035 * max(dlon, dlat)
    return float(np.clip(r, 0.004, 0.02))


def blob(lon_k, lat_k, r):
    """Poligono que envolve os pontos do frame (concave hull suavizado).

    O concave hull contem TODOS os pontos; um buffer pequeno arredonda as
    bordas e garante folga, deixando com cara de mancha.
    """
    mask = np.isfinite(lon_k) & np.isfinite(lat_k)
    pts = np.column_stack([lon_k[mask], lat_k[mask]])
    if len(pts) < 3:
        if len(pts) == 0:
            return None
        return MultiPoint([tuple(p) for p in pts]).buffer(r)
    mp = MultiPoint([tuple(p) for p in pts])
    try:
        geom = concave_hull(mp, ratio=0.4)
    except Exception:
        geom = mp.convex_hull
    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        geom = mp.convex_hull
    # arredonda cantos e dá folga (nada de pontos fora)
    geom = geom.buffer(r * 0.25).buffer(-r * 0.1)
    return geom


def plot_geom(ax, geom, color):
    if geom is None or geom.is_empty:
        return
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for poly in polys:
        verts, codes = [], []
        for ring in [poly.exterior, *poly.interiors]:
            xy = np.asarray(ring.coords)
            if len(xy) < 3:
                continue
            verts.extend(xy)
            codes.append(MplPath.MOVETO)
            codes.extend([MplPath.LINETO] * (len(xy) - 2))
            codes.append(MplPath.CLOSEPOLY)
        if not verts:
            continue
        path = MplPath(verts, codes)
        ax.add_patch(PathPatch(path, facecolor=color, edgecolor=color,
                               lw=1.2, alpha=0.35))


def compute_extent(fc, hc, r):
    lon = np.concatenate([fc[0].ravel(), hc[0].ravel()])
    lat = np.concatenate([fc[1].ravel(), hc[1].ravel()])
    lon_min, lon_max = np.nanmin(lon), np.nanmax(lon)
    lat_min, lat_max = np.nanmin(lat), np.nanmax(lat)
    dlon = max(lon_max - lon_min, 1e-3)
    dlat = max(lat_max - lat_min, 1e-3)
    mlon = dlon * MARGIN_FRAC + r
    mlat = dlat * MARGIN_FRAC + r
    return (lon_min - mlon, lon_max + mlon, lat_min - mlat, lat_max + mlat)


def make_gif(key, tag, fc_path, hc_path):
    fc = load_xy(fc_path)[:2] + (None,)
    fc_lon, fc_lat, times = load_xy(fc_path)
    hc_lon, hc_lat, _ = load_xy(hc_path)
    fc = (fc_lon, fc_lat)
    hc = (hc_lon, hc_lat)
    r = buffer_radius(fc, hc)
    extent = compute_extent(fc, hc, r)
    n_time = fc_lon.shape[1]

    FRAME_TMP.mkdir(parents=True, exist_ok=True)
    frame_files = []
    for k in range(n_time):
        fig, ax = plt.subplots(figsize=(8, 7))
        # poligonos (manchas)
        plot_geom(ax, blob(fc_lon[:, k], fc_lat[:, k], r), FC_COLOR)
        plot_geom(ax, blob(hc_lon[:, k], hc_lat[:, k], r), HC_COLOR)
        # pontos discretos por cima, sutis, para dar textura
        ax.scatter(fc_lon[:, k], fc_lat[:, k], s=2, c=FC_COLOR, alpha=0.25, edgecolors="none")
        ax.scatter(hc_lon[:, k], hc_lat[:, k], s=2, c=HC_COLOR, alpha=0.25, edgecolors="none")
        # itens de legenda
        ax.scatter([], [], s=40, c=FC_COLOR, alpha=0.5, label="Forecast")
        ax.scatter([], [], s=40, c=HC_COLOR, alpha=0.5, label="Hindcast")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, alpha=0.3)
        ts = np.datetime_as_string(times[k], unit="m").replace("T", " ")
        ax.set_title(f"{key.upper()} — {tag} | {ts}\nForecast (azul) x Hindcast (vermelho)")
        ax.legend(loc="upper right", markerscale=1)
        frame = FRAME_TMP / f"{key}_{k:03d}.png"
        fig.savefig(frame, dpi=110, bbox_inches="tight")
        plt.close(fig)
        frame_files.append(frame)

    imgs = [imageio.imread(str(f)) for f in frame_files]
    h = min(im.shape[0] for im in imgs)
    w = min(im.shape[1] for im in imgs)
    imgs = [im[:h, :w] for im in imgs]
    out = GIF_DIR / f"{key}_{tag}_mancha.gif"
    imageio.mimsave(str(out), imgs, fps=FPS, loop=0)
    for f in frame_files:
        f.unlink(missing_ok=True)
    print(f"  {out.name}  (r={r:.4f}deg, {n_time} frames)")
    return out


def main():
    pairs = find_pairs()
    print(f"{len(pairs)} pares. Saida: {GIF_DIR}\n")
    for key, (tag, fc, hc) in pairs.items():
        print(f"== {key} ({tag})")
        make_gif(key, tag, fc, hc)
    if FRAME_TMP.exists():
        try:
            FRAME_TMP.rmdir()
        except OSError:
            pass
    print("\nConcluido.")


if __name__ == "__main__":
    main()
