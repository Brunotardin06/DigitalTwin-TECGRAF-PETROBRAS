import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from analiseVetorial import extract_components, _extract_lon_lat_coords, _extract_uv_components




def load_data(file, start_time, end_time):
    dataset  = xr.open_dataset(file)

    start = np.datetime64(start_time)
    end = np.datetime64(end_time)

    time_mask = (end >= dataset.time) & ( dataset.time >= start)
    data_slice = dataset.isel(time=time_mask)

    num_timesteps = len(data_slice.time)

    print(f"Intervalo solicitado: {start_time} até {end_time}")
    print(f"Quantidade de timesteps: {num_timesteps}")
    if num_timesteps > 0:
        t_start_real = str(data_slice.time.values[0]).replace('T', ' ').split('.')[0]
        t_end_real = str(data_slice.time.values[-1]).replace('T', ' ').split('.')[0]
        print(f"Intervalo real: {t_start_real} até {t_end_real}")

    return dataset, data_slice, num_timesteps
    
def extract_temporal_components(data_slice):
    times = data_slice.time.values
    results = []

    for i, t in enumerate(times):
        t_str = str(t).replace('T', ' ').split('.')[0]
        try:
            slice_t = data_slice.sel(time=t, method="nearest")
            u, v, u_name, v_name = _extract_uv_components(slice_t)
            lon, lat = _extract_lon_lat_coords(slice_t)

            u = u.squeeze(drop=True)
            v = v.squeeze(drop=True)

            results.append({
                'time':t,
                'time_str': t_str,
                'u': u,
                'v': v,
                'lon': lon,
                'lat': lat,
                'u_name': u_name,
                'v_name': v_name
            })
            print(f"  [{i+1}/{len(times)}] {t_str} - OK")
        except Exception as e:
            print(f"  [{i+1}/{len(times)}] {t_str} - ERRO: {e}")
    
    return results

def analyze_temporal_variation(results):
    if not results:
        return None
    
    u_list = [r['u'].values.flatten() for r in results]
    v_list = [r['v'].values.flatten() for r in results]

    stats_list = []
    for i, r in enumerate(results):
        u_flat = r['u'].values.flatten()
        v_flat = r['v'].values.flatten()

        u_valid = u_flat[~np.isnan(u_flat)]
        v_valid = v_flat[~np.isnan(v_flat)]

        if len(u_valid) > 0 and len(v_valid) > 0:
            magnitude = np.sqrt(u_valid**2 + v_valid**2)

            stats_list.append({
                'time_str': r['time_str'],
                'u_mean': np.mean(u_valid),
                'u_std': np.std(u_valid),
                'u_min': np.min(u_valid),
                'u_max': np.max(u_valid),
                'v_mean': np.mean(v_valid),
                'v_std': np.std(v_valid),
                'v_min': np.min(v_valid),
                'v_max': np.max(v_valid), 
                'mag_mean': np.mean(magnitude),
                'mag_std': np.std(magnitude),
                'mag_max': np.max(magnitude)
            })
    print("\nTIMESTEP | U_mean | U_std | V_mean | V_std | |V|_mean | |V|_max")
    for s in stats_list:
        print(f"{s['time_str']} | {s['u_mean']:+.6f} | {s['u_std']:.6f} | "
              f"{s['v_mean']:+.6f} | {s['v_std']:.6f} | {s['mag_mean']:.6f} | {s['mag_max']:.6f}")
    
    return {
        'results': results,
        'stats': stats_list
    }


def plot_temporal_evolution(analysis_data):
    if not analysis_data:
        print("Nenhum dado para plotar")
        return None
    
    stats = analysis_data['stats']
    times = np.arange(len(stats))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    time_labels = [s['time_str'] for s in stats]

    u_means = [s['u_mean'] for s in stats]
    u_stds = [s['u_std'] for s in stats]
    axes[0].errorbar(times, u_means, yerr=u_stds, marker='o', linestyle='-', color='red', alpha=0.7)
    axes[0].set_title('Evolução da Componente U', fontsize=12, weight='bold')
    axes[0].set_ylabel('u (m/s)')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color='black', linestyle='--', linewidth=0.5)
    
    # V component evolution
    v_means = [s['v_mean'] for s in stats]
    v_stds = [s['v_std'] for s in stats]
    axes[1].errorbar(times, v_means, yerr=v_stds, marker='o', linestyle='-', color='blue', alpha=0.7)
    axes[1].set_title('Evolução da Componente V', fontsize=12, weight='bold')
    axes[1].set_ylabel('v (m/s)')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0, color='black', linestyle='--', linewidth=0.5)
    
    # Standard deviation evolution
    u_stds_vals = [s['u_std'] for s in stats]
    v_stds_vals = [s['v_std'] for s in stats]
    axes[2].plot(times, u_stds_vals, marker='o', linestyle='-', color='red', label='σ(u)', alpha=0.7)
    axes[2].plot(times, v_stds_vals, marker='s', linestyle='-', color='blue', label='σ(v)', alpha=0.7)
    axes[2].set_title('Evolução do Desvio Padrão', fontsize=12, weight='bold')
    axes[2].set_ylabel('Desvio Padrão (m/s)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Set x-axis labels
    for ax in axes.flat:
        ax.set_xticks(times[::max(1, len(times)//6)])
        ax.set_xticklabels([time_labels[i] for i in ax.get_xticks() if i < len(time_labels)], 
                           rotation=45, ha='right')
    
    plt.suptitle('Análise Temporal das Componentes de Velocidade', fontsize=14, weight='bold')
    plt.tight_layout()
    
    return fig

def main():
    file = r'C:\Users\prmorais\Documents\GitHub\oil-drift-simulations\analysis\modelos\corrente_cmems_mod_glo_phy_anfc_0.083deg_RUN_2025-06-12.nc'
    start_time = '2025-06-12T00:00:00'
    end_time = '2025-06-13T00:00:00'

    # 1. Carrega dados
    dataset, data_slice, num_timesteps = load_data(file, start_time, end_time)
    
    if num_timesteps == 0:
        print("Nenhum timestep encontrado no intervalo")
        return
    
    # 2. Extrai componentes para todos os timesteps
    results = extract_temporal_components(data_slice)
    
    if not results:
        print("Falha ao extrair componentes")
        return
    
    # 3. Analisa variação temporal
    analysis_data = analyze_temporal_variation(results)
    
    # 4. Plota evolução temporal
    fig = plot_temporal_evolution(analysis_data)
    
    plt.show()


if __name__ == "__main__":
    main()