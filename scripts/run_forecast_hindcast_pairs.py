"""
Roda 10 PARES de simulacoes de deriva de oleo (forecast x hindcast).

Baseado no motor de simulacao do projeto deriva-main (src/simulation.py:
seed por ponto + data + duracao), adaptado para:

  * Cada PAR = duas simulacoes IDENTICAS (mesmo ponto, mesma data, mesmos
    parametros e MESMA semente aleatoria), variando apenas o dataset de
    correnteza: forecast vs hindcast.
  * 10 datas distintas distribuidas pelas coberturas dos pares de correnteza
    (abr/2024, out/2024, jun/2025).
  * Sem vento (fallback 0): vento/sal-temp so existem para jun/2025, e rodar
    so com corrente isola exatamente a diferenca forecast x hindcast.

Saidas: data/2-simulated/forecast_hindcast_pairs/
"""
from datetime import datetime, timedelta
from pathlib import Path
import random

import numpy as np
from opendrift.models.openoil import OpenOil
from opendrift.readers import reader_netCDF_CF_generic

ROOT = Path(__file__).resolve().parents[1]
MC = ROOT / "data" / "1-raw" / "marine_copernicus"
FC_CF = ROOT / "data" / "cache" / "forecast_cf"
OUT_DIR = ROOT / "data" / "2-simulated" / "forecast_hindcast_pairs"

# --- ponto de semeadura (offshore, P-67; dentro da cobertura dos 3 pares) ---
SEED_LON = -42.726
SEED_LAT = -25.272

# --- parametros de simulacao (leves, estilo sim4validation) ---
NUM_SEED_ELEMENTS = 2000
SEED_RADIUS_MEAN_M = 2000.0
SEED_RADIUS_STD_M = 500.0
SEED_RADIUS_MIN_M = 1000.0
MAX_LEAK_DURATION_HOURS = 12
DURATION_DAYS = 1
TIME_STEP_MINUTES = 5
OUTPUT_TIME_STEP_MINUTES = 60
OIL_TYPE = "SOCKEYE SWEET"
EXPORT_VARIABLES = ["water_content"]

# --- as 10 datas (distintas) e a qual par de correnteza pertencem ---
# tag = mes do par (arquivos forecast_/hindcast_br_se_<tag>.nc)
DATES = [
    ("2024_04", datetime(2024, 4, 5)),
    ("2024_04", datetime(2024, 4, 14)),
    ("2024_04", datetime(2024, 4, 23)),
    ("2024_10", datetime(2024, 10, 6)),
    ("2024_10", datetime(2024, 10, 15)),
    ("2024_10", datetime(2024, 10, 24)),
    ("2025_06", datetime(2025, 6, 5)),
    ("2025_06", datetime(2025, 6, 12)),
    ("2025_06", datetime(2025, 6, 19)),
    ("2025_06", datetime(2025, 6, 26)),
]


def current_path(source, tag):
    if source == "forecast":
        return FC_CF / f"forecast_br_se_{tag}_cf.nc"
    return MC / f"hindcast_br_se_{tag}.nc"


def run_one(source, tag, base_date, pair_index, draws):
    """Uma simulacao. `draws` fixa a aleatoriedade -> par identico."""
    o = OpenOil(loglevel=30)
    o.add_reader(reader_netCDF_CF_generic.Reader(str(current_path(source, tag))))
    # sem reader de vento -> usa constante 0 (nao aborta)
    o.set_config("environment:fallback:x_wind", 0)
    o.set_config("environment:fallback:y_wind", 0)

    seed_time_start = base_date + timedelta(hours=draws["start_hour"])
    seed_time_end = seed_time_start + timedelta(hours=draws["leak_hours"])
    radius = max(draws["radius"], SEED_RADIUS_MIN_M)

    o.seed_elements(
        lon=SEED_LON,
        lat=SEED_LAT,
        time=[seed_time_start, seed_time_end],
        number=NUM_SEED_ELEMENTS,
        radius=radius,
        oil_type=OIL_TYPE,
    )

    sim_end = seed_time_start + timedelta(days=DURATION_DAYS)
    out_name = (
        f"pair{pair_index:02d}_{tag}_{source}_"
        f"{seed_time_start.strftime('%Y%m%dT%H%M')}_{sim_end.strftime('%Y%m%dT%H%M')}.nc"
    )
    out_path = OUT_DIR / out_name
    o.run(
        duration=timedelta(days=DURATION_DAYS),
        time_step=TIME_STEP_MINUTES * 60,
        time_step_output=OUTPUT_TIME_STEP_MINUTES * 60,
        outfile=str(out_path),
        export_variables=EXPORT_VARIABLES,
    )
    return out_path, seed_time_start, sim_end


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saida: {OUT_DIR}\n")
    for i, (tag, base_date) in enumerate(DATES, start=1):
        # sorteios determinísticos por par -> forecast e hindcast identicos
        rng = random.Random(i)
        nrng = np.random.RandomState(i)
        draws = {
            "start_hour": rng.randint(0, 23),
            "leak_hours": rng.randint(1, MAX_LEAK_DURATION_HOURS),
            "radius": float(nrng.normal(SEED_RADIUS_MEAN_M, SEED_RADIUS_STD_M)),
        }
        print(f"=== PAR {i:02d} | {tag} | base {base_date.date()} | draws={draws}")
        for source in ("forecast", "hindcast"):
            out_path, t0, t1 = run_one(source, tag, base_date, i, draws)
            print(f"    {source:8s} {t0} -> {t1}  =>  {out_path.name}")
    print("\nConcluido: 10 pares (20 simulacoes).")


if __name__ == "__main__":
    main()
