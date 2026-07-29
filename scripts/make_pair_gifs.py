"""
Gera 1 GIF por par (forecast x hindcast) mostrando o movimento das duas
manchas sobrepostas. A extensao do mapa e fixada para conter TODAS as
particulas dos dois datasets em todos os instantes (com margem), de modo
que nenhuma parte da mancha e cortada em nenhum frame.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "2-simulated" / "forecast_hindcast_pairs"
GIF_DIR = D / "gifs"
GIF_DIR.mkdir(parents=True, exist_ok=True)
FRAME_TMP = GIF_DIR / "_frames"

FC_COLOR = "#1f77b4"   # forecast = azul
HC_COLOR = "#d62728"   # hindcast = vermelho
FPS = 4
MARGIN_FRAC = 0.08     # 8% de folga em volta -> garante nao cortar


def find_pairs():
    pairs = {}
    for f in sorted(D.glob("pair*_forecast_*.nc")):
        m = re.match(r"(pair\d+)_(\d{4}_\d{2})_forecast_(\d+T\d+)_(\d+T\d+)\.nc", f.name)
        if not m:
            continue
        key, tag, t0, t1 = m.groups()
        hc = f.with_name(f.name.replace("_forecast_", "_hindcast_"))
        if hc.exists():
            pairs[key] = (tag, t0, t1, f, hc)
    return dict(sorted(pairs.items()))


def load_xy(path):
    ds = xr.open_dataset(path)
    lon = ds["lon"].values.astype(float)  # (trajectory, time)
    lat = ds["lat"].values.astype(float)
    times = ds["time"].values
    ds.close()
    return lon, lat, times


def compute_extent(*arrays_lon_lat):
    lons = np.concatenate([a[0].ravel() for a in arrays_lon_lat])
    lats = np.concatenate([a[1].ravel() for a in arrays_lon_lat])
    lon_min, lon_max = np.nanmin(lons), np.nanmax(lons)
    lat_min, lat_max = np.nanmin(lats), np.nanmax(lats)
    dlon = max(lon_max - lon_min, 1e-3)
    dlat = max(lat_max - lat_min, 1e-3)
    mlon = dlon * MARGIN_FRAC
    mlat = dlat * MARGIN_FRAC
    return (lon_min - mlon, lon_max + mlon, lat_min - mlat, lat_max + mlat)


def make_gif(key, tag, t0, t1, fc_path, hc_path):
    fc_lon, fc_lat, times = load_xy(fc_path)
    hc_lon, hc_lat, _ = load_xy(hc_path)
    extent = compute_extent((fc_lon, fc_lat), (hc_lon, hc_lat))
    n_time = fc_lon.shape[1]

    FRAME_TMP.mkdir(parents=True, exist_ok=True)
    frame_files = []
    for k in range(n_time):
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(fc_lon[:, k], fc_lat[:, k], s=6, c=FC_COLOR, alpha=0.5,
                   label="Forecast", edgecolors="none")
        ax.scatter(hc_lon[:, k], hc_lat[:, k], s=6, c=HC_COLOR, alpha=0.5,
                   label="Hindcast", edgecolors="none")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, alpha=0.3)
        ts = np.datetime_as_string(times[k], unit="m").replace("T", " ")
        ax.set_title(f"{key.upper()} — {tag} | {ts}\nForecast (azul) x Hindcast (vermelho)")
        ax.legend(loc="upper right", markerscale=2)
        frame = FRAME_TMP / f"{key}_{k:03d}.png"
        fig.savefig(frame, dpi=110, bbox_inches="tight")
        plt.close(fig)
        frame_files.append(frame)

    # padroniza tamanho dos frames (bbox_inches pode variar 1px) recarregando
    imgs = [imageio.imread(str(f)) for f in frame_files]
    h = min(im.shape[0] for im in imgs)
    w = min(im.shape[1] for im in imgs)
    imgs = [im[:h, :w] for im in imgs]
    out = GIF_DIR / f"{key}_{tag}_compare.gif"
    imageio.mimsave(str(out), imgs, fps=FPS, loop=0)
    for f in frame_files:
        f.unlink(missing_ok=True)
    print(f"  {out.name}  ({n_time} frames, extent lon[{extent[0]:.3f},{extent[1]:.3f}] lat[{extent[2]:.3f},{extent[3]:.3f}])")
    return out


def main():
    pairs = find_pairs()
    print(f"{len(pairs)} pares encontrados. Saida: {GIF_DIR}\n")
    for key, (tag, t0, t1, fc, hc) in pairs.items():
        print(f"== {key} ({tag})")
        make_gif(key, tag, t0, t1, fc, hc)
    if FRAME_TMP.exists():
        try:
            FRAME_TMP.rmdir()
        except OSError:
            pass
    print("\nConcluido.")


if __name__ == "__main__":
    main()
