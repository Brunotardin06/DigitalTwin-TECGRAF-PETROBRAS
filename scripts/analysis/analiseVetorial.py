import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as crt
from scipy import stats
import os
import io
from datetime import datetime

from PIL import Image


def _infer_lat_lon_dims(data_array):
    lat_candidates = ("latitude", "lat")
    lon_candidates = ("longitude", "lon")

    lat_dim = next((d for d in data_array.dims if d.lower() in lat_candidates), None)
    lon_dim = next((d for d in data_array.dims if d.lower() in lon_candidates), None)

    if lat_dim is None or lon_dim is None:
        raise ValueError(f"Não foi possível inferir dimensões lat/lon em {data_array.dims}")

    return lat_dim, lon_dim


def _extract_lon_lat_coords(slice_data):
    coord_names = set(slice_data.coords)

    if "longitude" in coord_names and "latitude" in coord_names:
        return slice_data.longitude, slice_data.latitude

    if "lon" in coord_names and "lat" in coord_names:
        return slice_data.lon, slice_data.lat

    lon_name = next((c for c in slice_data.coords if "lon" in c.lower()), None)
    lat_name = next((c for c in slice_data.coords if "lat" in c.lower()), None)

    if lon_name is None or lat_name is None:
        raise KeyError(f"Coordenadas não reconhecidas. Disponíveis: {sorted(coord_names)}")

    return slice_data[lon_name], slice_data[lat_name]


def _extract_uv_components(slice_data):
    vars_available = set(slice_data.data_vars)
    candidates = [
        ("uo", "vo"),
        ("sw_cur_u", "sw_cur_v"),
        ("u", "v"),
    ]

    for u_name, v_name in candidates:
        if u_name in vars_available and v_name in vars_available:
            return slice_data[u_name].squeeze(drop=True), slice_data[v_name].squeeze(drop=True), u_name, v_name

    raise KeyError(
        "Par de variáveis de corrente não reconhecido. "
        f"Disponíveis: {sorted(vars_available)}"
    )


def load_data(file_ref, file_cmp, datetime_str):
    # Loads NRT and MY datasets and selects the specified time instant
    # Args: file_ref (NetCDF path), file_cmp (NetCDF path), datetime_str (ISO format)
    # Returns: (nrt_dataset, my_dataset, nrt_slice, my_slice, actual_time_nrt, actual_time_my)
    nrt = xr.open_dataset(file_ref)
    my = xr.open_dataset(file_cmp)
    
    # Select time using nearest method
    nrt_slice = nrt.sel(time=datetime_str, method="nearest")
    my_slice = my.sel(time=datetime_str, method="nearest")
    
    # Get actual selected times as strings
    actual_time_nrt = str(nrt_slice.time.values).replace('T', ' ').split('.')[0]
    actual_time_my = str(my_slice.time.values).replace('T', ' ').split('.')[0]
    requested_time_str = datetime_str.replace('T', ' ')
    
    # Validation and warnings
    print("\n" + "=" * 60)
    print("VALIDAÇÃO DO TIMESTAMP SELECIONADO:")
    print("=" * 60)
    print(f"Timestamp solicitado: {requested_time_str}")
    print(f"Timestamp NRT real:   {actual_time_nrt}")
    print(f"Timestamp MY real:    {actual_time_my}")
    
    # Check if NRT and MY have same timestamp
    if actual_time_nrt == actual_time_my:
        print("NRT e MY estão no MESMO timestamp")
    else:
        print(f"AVISO: NRT e MY têm timestamps DIFERENTES")
    
    return nrt, my, nrt_slice, my_slice, actual_time_nrt, actual_time_my


def extract_components(slice_data):
    # Extracts u, v components and coordinates from a data slice
    # Args: slice_data (xarray dataset filtered by time)
    # Returns: (u, v, lon, lat)
    u, v, u_name, v_name = _extract_uv_components(slice_data)
    lon, lat = _extract_lon_lat_coords(slice_data)
    #print(f"Componentes detectados: {u_name}/{v_name}")
    
    return u, v, lon, lat

"""
def calculate_spatial_crop(lat, lon, lat_min_req, lat_max_req, lon_min_req, lon_max_req, n_expand=3):
    # Calculates indices for spatial crop based on center and expansion
    # Args: lat, lon (dataset coordinates), lat/lon_min/max_req (requested limits), n_expand (extra points, default: 3)
    # Returns: (lat_indices, lon_indices, lat_min, lat_max, lon_min, lon_max)
    lat_center = (lat_min_req + lat_max_req) / 2
    lon_center = (lon_min_req + lon_max_req) / 2

    print(f"Debug: {lat.values}" )
    print(f"Debug 3:  {lon.values}" )
    print(f"Debug 2:  {lat.values.shape}" )

    
    lat_center_idx = np.argmin(np.abs(lat.values - lat_center))
    lon_center_idx = np.argmin(np.abs(lon.values - lon_center))
    
    lat_min_idx = max(0, lat_center_idx - n_expand)
    lat_max_idx = min(len(lat.values) - 1, lat_center_idx + n_expand)
    lon_min_idx = max(0, lon_center_idx - n_expand)
    lon_max_idx = min(len(lon.values) - 1, lon_center_idx + n_expand)
    
    lat_indices = np.arange(lat_min_idx, lat_max_idx + 1)
    lon_indices = np.arange(lon_min_idx, lon_max_idx + 1)
    
    lat_min = lat.values[lat_min_idx]
    lat_max = lat.values[lat_max_idx]
    lon_min = lon.values[lon_min_idx]
    lon_max = lon.values[lon_max_idx]
    
    return lat_indices, lon_indices, lat_min, lat_max, lon_min, lon_max
"""

def calculate_spatial_crop(lat, lon, lat_min_req=None, lat_max_req=None, lon_min_req=None, lon_max_req=None, n_expand=3):
    # Calculates indices for spatial crop based on requested bounds and expansion.
    # Missing limits are treated as open bounds; if both bounds of an axis are missing,
    # the full axis is used.
    # Args: lat, lon (dataset coordinates), lat/lon_min/max_req (optional requested limits), n_expand (extra points, default: 3)
    # Returns: (lat_indices, lon_indices, lat_min, lat_max, lon_min, lon_max)

    lat_vals = lat.values
    lon_vals = lon.values
    n_expand = max(0, int(n_expand))

    def compute_axis_indices(axis_vals, req_min, req_max):
        if req_min is None and req_max is None:
            return np.arange(len(axis_vals))

        axis_min = float(np.min(axis_vals))
        axis_max = float(np.max(axis_vals))

        lo = axis_min if req_min is None else req_min
        hi = axis_max if req_max is None else req_max
        lo, hi = (lo, hi) if lo <= hi else (hi, lo)

        mask = (axis_vals >= lo) & (axis_vals <= hi)

        if mask.any():
            axis_in = np.where(mask)[0]
            min_idx = max(0, axis_in.min() - n_expand)
            max_idx = min(len(axis_vals) - 1, axis_in.max() + n_expand)
        else:
            # Fallback: nearest points to requested bounds
            min_idx = int(np.argmin(np.abs(axis_vals - lo)))
            max_idx = int(np.argmin(np.abs(axis_vals - hi)))
            if min_idx > max_idx:
                min_idx, max_idx = max_idx, min_idx
            min_idx = max(0, min_idx - n_expand)
            max_idx = min(len(axis_vals) - 1, max_idx + n_expand)

        return np.arange(min_idx, max_idx + 1)

    lat_indices = compute_axis_indices(lat_vals, lat_min_req, lat_max_req)
    lon_indices = compute_axis_indices(lon_vals, lon_min_req, lon_max_req)

    lat_crop_vals = lat_vals[lat_indices]
    lon_crop_vals = lon_vals[lon_indices]

    lat_min = float(np.min(lat_crop_vals))
    lat_max = float(np.max(lat_crop_vals))
    lon_min = float(np.min(lon_crop_vals))
    lon_max = float(np.max(lon_crop_vals))

    return lat_indices, lon_indices, lat_min, lat_max, lon_min, lon_max



def apply_crop(u, v, lat, lon, lat_indices, lon_indices):
    # Applies spatial crop to components and coordinates
    # Args: u, v (velocity components), lat, lon (coordinates), lat_indices, lon_indices (crop indices)
    # Returns: (u_crop, v_crop, lat_crop, lon_crop)
    lat_dim, lon_dim = _infer_lat_lon_dims(u)
    u_crop = u.isel({lat_dim: lat_indices, lon_dim: lon_indices})
    v_crop = v.isel({lat_dim: lat_indices, lon_dim: lon_indices})
    lat_crop = lat.isel({lat.dims[0]: lat_indices})
    lon_crop = lon.isel({lon.dims[0]: lon_indices})
    
    return u_crop, v_crop, lat_crop, lon_crop


def ensure_dimensions(u, v):
    # Ensures components have dimensions (latitude, longitude)
    # Args: u, v (velocity components)
    # Returns: (u, v) with correct dimensions
    lat_dim_u, lon_dim_u = _infer_lat_lon_dims(u)
    lat_dim_v, lon_dim_v = _infer_lat_lon_dims(v)

    if lat_dim_u != lat_dim_v or lon_dim_u != lon_dim_v:
        raise ValueError(f"Dimensões de u/v incompatíveis: u={u.dims}, v={v.dims}")

    if u.dims != (lat_dim_u, lon_dim_u):
        u = u.transpose(lat_dim_u, lon_dim_u)

    if v.dims != (lat_dim_v, lon_dim_v):
        v = v.transpose(lat_dim_v, lon_dim_v)
    
    return u, v


def align_grids(u_my, v_my, lon_my, lat_my, lon_ref, lat_ref):
    # Aligns MY grid to reference grid (NRT) through interpolation
    # Args: u_my, v_my (MY components), lon_my, lat_my (MY coordinates), lon_ref, lat_ref (reference coordinates)
    # Returns: (u_my_aligned, v_my_aligned)
    same_lon = np.array_equal(lon_ref.values, lon_my.values)
    same_lat = np.array_equal(lat_ref.values, lat_my.values)

    if not (same_lon and same_lat):
        lat_dim_my, lon_dim_my = _infer_lat_lon_dims(u_my)
        lon_target = xr.DataArray(lon_ref.values, dims=[lon_dim_my])
        lat_target = xr.DataArray(lat_ref.values, dims=[lat_dim_my])

        u_my_aligned = u_my.interp({lon_dim_my: lon_target, lat_dim_my: lat_target})
        v_my_aligned = v_my.interp({lon_dim_my: lon_target, lat_dim_my: lat_target})
    else:
        u_my_aligned = u_my
        v_my_aligned = v_my
    
    return u_my_aligned, v_my_aligned


def calculate_metrics(u_nrt, v_nrt, u_my, v_my):
    # Calculates RMSE and vector error metrics for u and v components
    # Args: u_nrt, v_nrt (NRT components), u_my, v_my (components to evaluate - MY)
    # Returns: dict with comprehensive metrics
    diff_u = u_nrt - u_my
    diff_v = v_nrt - v_my
    
    # Component-wise RMSE
    rmse_u = float(np.sqrt(np.nanmean((diff_u.values) ** 2)))
    rmse_v = float(np.sqrt(np.nanmean((diff_v.values) ** 2)))
    
    # Vector magnitude-based metrics
    # Magnitude of NRT vectors
    mag_nrt = np.sqrt(u_nrt.values**2 + v_nrt.values**2)
    # Magnitude of MY vectors
    mag_my = np.sqrt(u_my.values**2 + v_my.values**2)
    # Magnitude of difference vectors
    mag_diff = np.sqrt(diff_u.values**2 + diff_v.values**2)
    
    # Vector RMSE (based on magnitude of difference vectors)
    rmse_vector = float(np.sqrt(np.nanmean(mag_diff**2)))
    
    # Mean vector error (average magnitude of difference)
    mean_vector_error = float(np.nanmean(mag_diff))
    
    # Maximum vector error
    max_vector_error = float(np.nanmax(mag_diff))
    
    # Mean magnitude of NRT vectors (for context)
    mean_mag_nrt = float(np.nanmean(mag_nrt))
    
    # Relative error (percentage)
    relative_error_pct = (mean_vector_error / mean_mag_nrt * 100) if mean_mag_nrt > 0 else np.nan
    
    return {
        'rmse_u': rmse_u,
        'rmse_v': rmse_v,
        'rmse_vector': rmse_vector,
        'mean_vector_error': mean_vector_error,
        'max_vector_error': max_vector_error,
        'mean_mag_nrt': mean_mag_nrt,
        'relative_error_pct': relative_error_pct,
        'diff_u': diff_u,
        'diff_v': diff_v
    }


def plot_comparison(u_nrt, v_nrt, u_my, v_my, diff_u, diff_v,
                     lon, lat, lon_min, lon_max, lat_min, lat_max, datetime_str,
                     label_ref="NRT", label_cmp="MY"):
    # Plots comparison between NRT, MY and vector difference
    # Args: u_nrt, v_nrt (NRT components), u_my, v_my (MY aligned components), diff_u, diff_v (vector differences),
    #       lon, lat (coordinates), lon/lat_min/max (spatial limits), datetime_str (date/time string)
    # Returns: (fig, axes)
    lon2d, lat2d = np.meshgrid(lon.values, lat.values)
    
    fig, axes = plt.subplots(1, 3, figsize=(28, 10), 
                             subplot_kw={'projection': crt.PlateCarree()})
    
    skip = (slice(None, None, 1), slice(None, None, 1))
    
    # Subplot NRT
    ax_nrt = axes[0]
    ax_nrt.coastlines()
    ax_nrt.set_extent([lon_min - 0.1, lon_max + 0.1, lat_min - 0.1, lat_max + 0.1], 
                      crs=crt.PlateCarree())
    ax_nrt.quiver(lon2d[skip], lat2d[skip], u_nrt.values[skip], v_nrt.values[skip],
                  color="red", angles="xy", scale_units="xy", scale=1.0, width=0.004,
                  transform=crt.PlateCarree())
    gl1 = ax_nrt.gridlines(draw_labels=True, alpha=0.4, linestyle='--')
    gl1.top_labels = False
    gl1.right_labels = False
    ax_nrt.set_title(f'Dados {label_ref}', fontsize=14, weight='bold')
    
    # Subplot MY
    ax_my = axes[1]
    ax_my.coastlines()
    ax_my.set_extent([lon_min - 0.1, lon_max + 0.1, lat_min - 0.1, lat_max + 0.1], 
                     crs=crt.PlateCarree())
    ax_my.quiver(lon2d[skip], lat2d[skip], u_my.values[skip], v_my.values[skip],
                 color="blue", angles="xy", scale_units="xy", scale=1.0, width=0.004,
                 transform=crt.PlateCarree())
    gl2 = ax_my.gridlines(draw_labels=True, alpha=0.4, linestyle='--')
    gl2.top_labels = False
    gl2.right_labels = False
    ax_my.set_title(f'Dados {label_cmp}', fontsize=14, weight='bold')
    
    # Subplot Difference
    ax_diff = axes[2]
    ax_diff.coastlines()
    ax_diff.set_extent([lon_min - 0.1, lon_max + 0.1, lat_min - 0.1, lat_max + 0.1], 
                       crs=crt.PlateCarree())
    ax_diff.quiver(lon2d[skip], lat2d[skip], diff_u.values[skip], diff_v.values[skip],
                   color="green", angles="xy", scale_units="xy", scale=1.0, width=0.004,
                   transform=crt.PlateCarree())
    gl3 = ax_diff.gridlines(draw_labels=True, alpha=0.4, linestyle='--')
    gl3.top_labels = False
    gl3.right_labels = False
    ax_diff.set_title(f'Diferença ({label_ref} - {label_cmp})', fontsize=14, weight='bold')
    
    # Format datetime for display (replace T with space)
    datetime_display = datetime_str.replace("T", " ")
    
    plt.suptitle(
        f'Comparação Vetorial - Região Selecionada ({datetime_display})\n'
        f'{label_ref} vs {label_cmp}',
        fontsize=16,
        weight='bold'
    )
    plt.tight_layout()
    
    return fig, axes


def _prepare_comparison_fields(ref_slice, cmp_slice,
                               lat_indices_ref, lon_indices_ref,
                               lat_indices_cmp, lon_indices_cmp):
    # Prepares cropped/aligned fields and metrics for a pair of time slices
    # Args: ref_slice, cmp_slice (xarray slices), index arrays for ref/cmp crop
    # Returns: dict with fields, grid and metrics
    u_ref, v_ref, lon_ref, lat_ref = extract_components(ref_slice)
    u_cmp, v_cmp, lon_cmp, lat_cmp = extract_components(cmp_slice)

    u_ref, v_ref, lat_ref_c, lon_ref_c = apply_crop(
        u_ref, v_ref, lat_ref, lon_ref, lat_indices_ref, lon_indices_ref
    )
    u_cmp, v_cmp, lat_cmp_c, lon_cmp_c = apply_crop(
        u_cmp, v_cmp, lat_cmp, lon_cmp, lat_indices_cmp, lon_indices_cmp
    )

    u_ref, v_ref = ensure_dimensions(u_ref, v_ref)
    u_cmp, v_cmp = ensure_dimensions(u_cmp, v_cmp)

    u_cmp_aligned, v_cmp_aligned = align_grids(
        u_cmp, v_cmp, lon_cmp_c, lat_cmp_c, lon_ref_c, lat_ref_c
    )

    metrics = calculate_metrics(u_ref, v_ref, u_cmp_aligned, v_cmp_aligned)

    return {
        'u_ref': u_ref,
        'v_ref': v_ref,
        'u_cmp_aligned': u_cmp_aligned,
        'v_cmp_aligned': v_cmp_aligned,
        'diff_u': metrics['diff_u'],
        'diff_v': metrics['diff_v'],
        'lon_ref': lon_ref_c,
        'lat_ref': lat_ref_c,
        'metrics': metrics,
    }


def generate_timestamp_evolution_gif(ref_dataset, cmp_dataset,
                                     lat_indices_ref, lon_indices_ref,
                                     lat_indices_cmp, lon_indices_cmp,
                                     lon_min, lon_max, lat_min, lat_max,
                                     label_ref, label_cmp,
                                     fixed_cmp_time_str,
                                     gif_start_time_str=None,
                                     gif_end_time_str=None,
                                     output_gif_path=None,
                                     fps=2,
                                     frame_stride=1,
                                     max_frames=None):
    # Generates a GIF by iterating reference timestamps while keeping comparison time fixed
    # Args: datasets, crop indices, spatial bounds, labels, fixed cmp time and optional start/end window
    # Returns: output_gif_path (str) or None
    if output_gif_path is None:
        output_gif_path = get_data_path(os.path.join('gifs', 'evolucao_temporal.gif'))

    def ensure_unique_output_path(path):
        if not os.path.exists(path):
            return path

        base, ext = os.path.splitext(path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = f"{base}_{timestamp}{ext}"
        suffix = 1

        while os.path.exists(candidate):
            candidate = f"{base}_{timestamp}_{suffix:02d}{ext}"
            suffix += 1

        return candidate

    output_dir = os.path.dirname(output_gif_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    output_gif_path = ensure_unique_output_path(output_gif_path)

    cmp_fixed_slice = cmp_dataset.sel(time=fixed_cmp_time_str, method="nearest")
    fixed_cmp_time_real = str(cmp_fixed_slice.time.values).replace('T', ' ').split('.')[0]

    times_ref = ref_dataset.time.values
    if gif_start_time_str is not None:
        start_ts = np.datetime64(gif_start_time_str)
        times_ref = times_ref[times_ref >= start_ts]
    if gif_end_time_str is not None:
        end_ts = np.datetime64(gif_end_time_str)
        times_ref = times_ref[times_ref <= end_ts]

    if len(times_ref) == 0:
        print("Nenhum timestamp encontrado no intervalo solicitado para gerar GIF.")
        return None

    frame_stride = max(1, int(frame_stride))
    times_ref = times_ref[::frame_stride]

    if max_frames is not None:
        max_frames = max(1, int(max_frames))
        times_ref = times_ref[:max_frames]

    if len(times_ref) == 0:
        print("Nenhum frame restante após aplicar stride/max_frames.")
        return None

    print("\n" + "=" * 60)
    print("GERAÇÃO DE GIF TEMPORAL")
    print("=" * 60)
    print(f"Modelo fixo ({label_cmp}): {fixed_cmp_time_real}")
    print(f"Frames ({label_ref} variando): {len(times_ref)}")
    print(f"Configuração: fps={fps}, frame_stride={frame_stride}, max_frames={max_frames}")
    print(f"Saída GIF: {output_gif_path}")

    frames = []
    frame_buffers = []

    for i, time_ref in enumerate(times_ref):
        ref_slice = ref_dataset.sel(time=time_ref, method="nearest")
        ref_time_real = str(ref_slice.time.values).replace('T', ' ').split('.')[0]

        prepared = _prepare_comparison_fields(
            ref_slice, cmp_fixed_slice,
            lat_indices_ref, lon_indices_ref,
            lat_indices_cmp, lon_indices_cmp
        )

        datetime_display = f"{label_ref}: {ref_time_real} | {label_cmp} fixo: {fixed_cmp_time_real}"

        fig, _ = plot_comparison(
            prepared['u_ref'], prepared['v_ref'],
            prepared['u_cmp_aligned'], prepared['v_cmp_aligned'],
            prepared['diff_u'], prepared['diff_v'],
            prepared['lon_ref'], prepared['lat_ref'],
            lon_min, lon_max, lat_min, lat_max,
            datetime_display,
            label_ref=label_ref,
            label_cmp=label_cmp
        )

        fig.text(
            0.5, 0.01,
            f"RMSE vetorial: {prepared['metrics']['rmse_vector']:.4f} m/s | "
            f"Erro médio vetorial: {prepared['metrics']['mean_vector_error']:.4f} m/s",
            ha='center', va='bottom', fontsize=10
        )

        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
        plt.close(fig)

        buffer.seek(0)
        frame = Image.open(buffer).convert('P', palette=Image.ADAPTIVE)
        frames.append(frame)
        frame_buffers.append(buffer)

        print(f"  [{i+1:3d}/{len(times_ref)}] frame gerado: {ref_time_real}")

    duration_ms = max(1, int(1000 / max(1, fps)))
    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False
    )

    for frame in frames:
        frame.close()
    for buffer in frame_buffers:
        buffer.close()

    print("=" * 60)
    print(f"GIF gerado com sucesso: {output_gif_path}")
    print("=" * 60)

    return output_gif_path

def analyze_error_distribution(diff_u, diff_v):
    # Analyzes error distribution to explain bin frequency patterns
    # Args: diff_u, diff_v (difference arrays)
    # Returns: None (prints analysis    
    # Flatten and remove NaN
    diff_u_flat = diff_u.values.flatten()
    diff_v_flat = diff_v.values.flatten()
    diff_u_flat = diff_u_flat[~np.isnan(diff_u_flat)]
    diff_v_flat = diff_v_flat[~np.isnan(diff_v_flat)]
    
    print("\n" + "=" * 60)
    print("ANÁLISE DA DISTRIBUIÇÃO DE ERROS:")
    print("=" * 60)
    
    # 1. Simetria
    print("\n1. SIMETRIA DOS DADOS:")
    skew_u = stats.skew(diff_u_flat)
    skew_v = stats.skew(diff_v_flat)
    print(f"  Assimetria (Skewness) u: {skew_u:.6f}")
    print(f"  Assimetria (Skewness) v: {skew_v:.6f}")
    if abs(skew_u) < 0.5:
        print(f"    → u é aproximadamente SIMÉTRICO")
    elif skew_u > 0:
        print(f"    → u tem ASSIMETRIA POSITIVA (cauda direita)")
    else:
        print(f"    → u tem ASSIMETRIA NEGATIVA (cauda esquerda)")
    
    # 2. Curtose
    print("\n2. CURTOSE (CONCENTRAÇÃO NOS EXTREMOS):")
    kurt_u = stats.kurtosis(diff_u_flat)
    kurt_v = stats.kurtosis(diff_v_flat)
    print(f"  Curtose u: {kurt_u:.6f}")
    print(f"  Curtose v: {kurt_v:.6f}")
    if abs(kurt_u) < 0.5:
        print(f"    → u segue distribuição similar à GAUSSIANA")
    elif kurt_u > 0:
        print(f"    → u tem PICOS ACENTUADOS (leptocúrtica)")
    else:
        print(f"    → u é mais ACHATADA (platicúrtica)")
    
    # 3. Correlação
    print("\n3. CORRELAÇÃO ENTRE u E v:")
    corr = np.corrcoef(diff_u_flat, diff_v_flat)[0, 1]
    print(f"  Correlação de Pearson: {corr:.6f}")
    if abs(corr) < 0.3:
        print(f"    → u e v são INDEPENDENTES")
    elif abs(corr) < 0.7:
        print(f"    → u e v têm CORRELAÇÃO MODERADA")
    else:
        print(f"    → u e v são ALTAMENTE CORRELACIONADOS")
    
    # 4. Valores Únicos
    print("\n4. PRECISÃO NUMÉRICA:")
    unique_u = len(np.unique(diff_u_flat))
    unique_v = len(np.unique(diff_v_flat))
    print(f"  Valores únicos em u: {unique_u} de {len(diff_u_flat)} (densidade: {unique_u/len(diff_u_flat):.2%})")
    print(f"  Valores únicos em v: {unique_v} de {len(diff_v_flat)} (densidade: {unique_v/len(diff_v_flat):.2%})")
    if unique_u < len(diff_u_flat) * 0.5:
        print(f"    → u tem VALORES REPETIDOS (arredondamento?)")
    if unique_v < len(diff_v_flat) * 0.5:
        print(f"    → v tem VALORES REPETIDOS (arredondamento?)")
    
    # 5. Distribuição de Valores
    print("\n5. DISTRIBUIÇÃO DE FREQUÊNCIA DE VALORES:")
    value_counts_u = {}
    for val in diff_u_flat:
        val_round = round(val, 8)
        value_counts_u[val_round] = value_counts_u.get(val_round, 0) + 1
    
    top_values_u = sorted(value_counts_u.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Top 5 valores mais frequentes em u:")
    for val, count in top_values_u:
        print(f"    {val:.8f}: {count} vezes")

def analyze_gaussian_fit(diff_u, diff_v):
    # Analyzes if error distribution follows a Gaussian (normal) distribution
    # Performs normality tests and displays ideal Gaussian parameters
    # Args: diff_u, diff_v (difference arrays)
    # Returns: None (prints analysis and generates Q-Q plots)
    
    # Flatten and remove NaN
    diff_u_flat = diff_u.values.flatten()
    diff_v_flat = diff_v.values.flatten()
    diff_u_flat = diff_u_flat[~np.isnan(diff_u_flat)]
    diff_v_flat = diff_v_flat[~np.isnan(diff_v_flat)]
    
    print("\n" + "=" * 60)
    print("TESTE DE DISTRIBUIÇÃO GAUSSIANA:")
    print("=" * 60)
    
    # Calculate parameters
    mean_u = np.mean(diff_u_flat)
    std_u = np.std(diff_u_flat)
    mean_v = np.mean(diff_v_flat)
    std_v = np.std(diff_v_flat)
    
    # Display parameters
    print("\nPARÂMETROS DOS DADOS:")
    print(f"\n  Componente U:")
    print(f"    Media (mu):     {mean_u:+.8f}")
    print(f"    Desvio (sigma): {std_u:.8f}")
    
    print(f"\n  Componente V:")
    print(f"    Media (mu):     {mean_v:+.8f}")
    print(f"    Desvio (sigma): {std_v:.8f}")
    
    # Shapiro-Wilk Test
    
    stat_u_sw, p_u_sw = stats.shapiro(diff_u_flat)
    stat_v_sw, p_v_sw = stats.shapiro(diff_v_flat)
    
    result_u_sw = "GAUSSIANA" if p_u_sw > 0.05 else "NAO GAUSSIANA"
    result_v_sw = "GAUSSIANA" if p_v_sw > 0.05 else "NAO GAUSSIANA"
    
    print(f"\n  Componente U: p-value = {p_u_sw:.6f}  [{result_u_sw}]")
    print(f"  Componente V: p-value = {p_v_sw:.6f}  [{result_v_sw}]")
    
    # Summary
    print("\n" + "=" * 60)
    
    fig_qq, axes_qq = plt.subplots(1, 2, figsize=(12, 4.5))
    fig_qq.suptitle('Q-Q Plot: Comparação com Distribuição Gaussiana', fontsize=14, weight='bold')
    
    stats.probplot(diff_u_flat, dist="norm", plot=axes_qq[0])
    axes_qq[0].set_title('Componente U', fontsize=12, weight='bold')
    axes_qq[0].set_xlabel('Quantis Teóricos', fontsize=10)
    axes_qq[0].set_ylabel('Quantis Observados', fontsize=10)
    axes_qq[0].grid(True, alpha=0.3)
    
    stats.probplot(diff_v_flat, dist="norm", plot=axes_qq[1])
    axes_qq[1].set_title('Componente V', fontsize=12, weight='bold')
    axes_qq[1].set_xlabel('Quantis Teóricos', fontsize=10)
    axes_qq[1].set_ylabel('Quantis Observados', fontsize=10)
    axes_qq[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    print("Q-Q plots gerados (pontos na diagonal = gaussiana)")
    print("=" * 60)
    
    return fig_qq

def plot_histogramdif(diff_u, diff_v, u_my, v_my):
    # Plots histograms for u, v component differences and vector magnitude difference
    # Normalizes by maximum magnitude of MY vectors
    # Args: diff_u, diff_v (difference arrays), u_my, v_my (MY velocity components)
    # Returns: fig
    
    # Calculate magnitude of MY vectors
    mag_my = np.sqrt(u_my.values**2 + v_my.values**2)
    mag_my_max = float(np.nanmax(mag_my))
    
    print(f"\nMagnitude máxima dos vetores MY: {mag_my_max:.6f} m/s")
    
    # Flatten arrays -> To change a multidimensional array into a 1D array for histogram plotting
    diff_u_flat = diff_u.values.flatten()
    diff_v_flat = diff_v.values.flatten()
    
    # Normalize by maximum MY magnitude
    diff_u_norm = diff_u_flat / mag_my_max
    diff_v_norm = diff_v_flat / mag_my_max
    
    # Calculate vector magnitude difference (normalized)
    mag_diff_norm = np.sqrt(diff_u_norm**2 + diff_v_norm**2)
    
    # Remove NaN values
    diff_u_norm = diff_u_norm[~np.isnan(diff_u_norm)]
    diff_v_norm = diff_v_norm[~np.isnan(diff_v_norm)]
    mag_diff_norm = mag_diff_norm[~np.isnan(mag_diff_norm)]
    
    # Calculate optimal bins using Sturges' rule: k = ceil(log2(n) + 1)
    n_points = len(mag_diff_norm)
    n_bins = int(np.ceil(np.log2(n_points) + 1))
    
    print("\n" + "=" * 60)
    print("INFORMAÇÕES DOS HISTOGRAMAS:")
    print("=" * 60)
    print(f"Número de pontos: {n_points}")
    print(f"Número de bins (Regra de Sturges): {n_bins}")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Histogram diff_u (normalized)
    counts_u, bins_u, patches_u = axes[0].hist(diff_u_norm, bins=n_bins, color='red', alpha=0.7, edgecolor='black')
    axes[0].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Diferença u normalizada (adimensional)', fontsize=12)
    axes[0].set_ylabel('Frequência', fontsize=12)
    axes[0].set_title('Histograma Diferença u (Normalizada)', fontsize=14, weight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Print bin edges for diff_u
    print("\nBins Diferença u:")
    for i in range(len(counts_u)):
        left_bracket = "["
        right_bracket = "]" if i == len(counts_u) - 1 else ")"
        print(f"  Bin {i+1}: {left_bracket}{bins_u[i]:.6f}, {bins_u[i+1]:.6f}{right_bracket} → {int(counts_u[i])} valores")
    
    # Histogram diff_v (normalized)
    counts_v, bins_v, patches_v = axes[1].hist(diff_v_norm, bins=n_bins, color='blue', alpha=0.7, edgecolor='black')
    axes[1].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Diferença v normalizada (adimensional)', fontsize=12)
    axes[1].set_ylabel('Frequência', fontsize=12)
    axes[1].set_title('Histograma Diferença v (Normalizada)', fontsize=14, weight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Print bin edges for diff_v
    print("\nBins Diferença v:")
    for i in range(len(counts_v)):
        left_bracket = "["
        right_bracket = "]" if i == len(counts_v) - 1 else ")"
        print(f"  Bin {i+1}: {left_bracket}{bins_v[i]:.6f}, {bins_v[i+1]:.6f}{right_bracket} → {int(counts_v[i])} valores")
    
    # Histogram vector magnitude (normalized)
    counts_mag, bins_mag, patches_mag = axes[2].hist(mag_diff_norm, bins=n_bins, color='green', alpha=0.7, edgecolor='black')
    axes[2].set_xlabel('Magnitude da Diferença Normalizada (adimensional)', fontsize=12)
    axes[2].set_ylabel('Frequência', fontsize=12)
    axes[2].set_title('Histograma Erro Vetorial (Normalizado)', fontsize=14, weight='bold')
    axes[2].grid(True, alpha=0.3)
    
    # Print bin edges for magnitude
    print("\nBins Magnitude Vetorial:")
    for i in range(len(counts_mag)):
        left_bracket = "["
        right_bracket = "]" if i == len(counts_mag) - 1 else ")"
        print(f"  Bin {i+1}: {left_bracket}{bins_mag[i]:.6f}, {bins_mag[i+1]:.6f}{right_bracket} → {int(counts_mag[i])} valores")
    print("=" * 60)
    
    plt.suptitle(f'Distribuição dos Erros Normalizados (NRT - MY) | Norm: mag_MY_max = {mag_my_max:.4f} m/s', fontsize=16, weight='bold')
    plt.tight_layout()
    
    return fig


def analyze_all_timestamps_distribution(nrt, my, lat_indices_nrt, lon_indices_nrt, lat_indices_my, lon_indices_my):
    # Iterates over all NRT timestamps and analyses if NRT-MY spatial differences
    # follow a Gaussian distribution for each instant.
    # Args: nrt, my (full xarray datasets), lat_indices, lon_indices (pre-computed crop indices)
    # Returns: list of dicts with per-timestamp statistics

    times_nrt = nrt.time.values
    results = []

    print("\n" + "=" * 60)
    print("ANÁLISE TEMPORAL DA DISTRIBUIÇÃO DAS DIFERENÇAS:")
    print("=" * 60)
    print(f"Iterando sobre {len(times_nrt)} timestamps\n")

    for i, t in enumerate(times_nrt):
        t_str = str(t).replace('T', ' ').split('.')[0]
        try:
            nrt_slice = nrt.sel(time=t, method="nearest")
            my_slice  = my.sel(time=t, method="nearest")

            u_nrt, v_nrt, lon_nrt, lat_nrt = extract_components(nrt_slice)
            u_my,  v_my,  lon_my,  lat_my  = extract_components(my_slice)

            # Apply same spatial crop
            u_nrt, v_nrt, lat_c,    lon_c    = apply_crop(u_nrt, v_nrt, lat_nrt, lon_nrt, lat_indices_nrt, lon_indices_nrt)
            u_my,  v_my,  lat_my_c, lon_my_c = apply_crop(u_my,  v_my,  lat_my,  lon_my,  lat_indices_my, lon_indices_my)

            # Ensure correct dimensions
            u_nrt, v_nrt = ensure_dimensions(u_nrt, v_nrt)
            u_my,  v_my  = ensure_dimensions(u_my,  v_my)

            # Align MY grid to NRT grid
            u_my_al, v_my_al = align_grids(u_my, v_my, lon_my_c, lat_my_c, lon_c, lat_c)

            # Flatten and remove NaN
            du = (u_nrt - u_my_al).values.flatten()
            dv = (v_nrt - v_my_al).values.flatten()
            du = du[~np.isnan(du)]
            dv = dv[~np.isnan(dv)]

            # Vector-by-vector magnitude variation (NRT vs MY)
            mag_nrt = np.sqrt(u_nrt.values.flatten()**2 + v_nrt.values.flatten()**2)
            mag_my = np.sqrt(u_my_al.values.flatten()**2 + v_my_al.values.flatten()**2)
            dmag = mag_nrt - mag_my
            mask_mag = ~(np.isnan(mag_nrt) | np.isnan(mag_my))
            dmag = dmag[mask_mag]

            # Vector-by-vector angle variation (NRT vs MY), wrapped to [-180, 180]
            theta_nrt = np.degrees(np.arctan2(v_nrt.values.flatten(), u_nrt.values.flatten()))
            theta_my = np.degrees(np.arctan2(v_my_al.values.flatten(), u_my_al.values.flatten()))
            dtheta = theta_nrt - theta_my
            dtheta = ((dtheta + 180.0) % 360.0) - 180.0
            mask_theta = ~(np.isnan(theta_nrt) | np.isnan(theta_my))
            dtheta = dtheta[mask_theta]
            dtheta_time_str = np.full(dtheta.shape, t_str, dtype=object)

            if len(du) < 3 or len(dv) < 3 or len(dmag) < 3:
                continue

            mean_u, std_u = float(np.mean(du)), float(np.std(du))
            mean_v, std_v = float(np.mean(dv)), float(np.std(dv))
            skew_u = float(stats.skew(du))
            skew_v = float(stats.skew(dv))
            kurt_u = float(stats.kurtosis(du))
            kurt_v = float(stats.kurtosis(dv))

            # Shapiro-Wilk: best for small samples (n < 5000)
            _, p_sw_u = stats.shapiro(du)
            _, p_sw_v = stats.shapiro(dv)

            results.append({
                'time': t,
                'time_str': t_str,
                'mean_u': mean_u, 'std_u': std_u,
                'mean_v': mean_v, 'std_v': std_v,
                'skew_u': skew_u, 'skew_v': skew_v,
                'kurt_u': kurt_u, 'kurt_v': kurt_v,
                'p_sw_u': p_sw_u, 'p_sw_v': p_sw_v,
                'diff_u': du,     'diff_v': dv,
                'diff_mag': dmag,
                'diff_theta_deg': dtheta,
                'diff_theta_time_str': dtheta_time_str,
                'n_points': len(du)
            })

            tag_u = "GAUSSIANA"     if p_sw_u > 0.05 else "NAO-GAUSSIANA"
            tag_v = "GAUSSIANA"     if p_sw_v > 0.05 else "NAO-GAUSSIANA"
            n_samples = len(du)
            print(f"  [{i+1:3d}/{len(times_nrt)}] {t_str} | n={n_samples} | "
                  f"μu={mean_u:+.4f} σu={std_u:.4f} p_sw={p_sw_u:.10f} [{tag_u}] | "
                  f"μv={mean_v:+.4f} σv={std_v:.4f} p_sw={p_sw_v:.10f} [{tag_v}]")

        except Exception as e:
            print(f"  [{i+1:3d}/{len(times_nrt)}] {t_str} - ERRO: {e}")

    total = len(results)
    if total > 0:
        n_gauss_u = sum(1 for r in results if r['p_sw_u'] > 0.05)
        n_gauss_v = sum(1 for r in results if r['p_sw_v'] > 0.05)
        print(f"\nRESUMO FINAL:")
        print(f"  U: {n_gauss_u}/{total} timestamps com distribuição Gaussiana "
              f"({100 * n_gauss_u / total:.1f}%)")
        print(f"  V: {n_gauss_v}/{total} timestamps com distribuição Gaussiana "
              f"({100 * n_gauss_v / total:.1f}%)")

    return results


def plot_all_timestamps_histogram(ts_results):
    # Aggregates differences from ALL timestamps into a single histogram per component
    # and checks if the combined distribution follows a Gaussian.
    # Uses Shapiro-Wilk for n <= 5000, D'Agostino-Pearson (normaltest) for larger samples.
    # Args: ts_results (list of dicts returned by analyze_all_timestamps_distribution)
    # Returns: fig
    if not ts_results:
        print("Nenhum resultado para plotar.")
        return None

    # Concatenate all timestamps into single arrays
    all_du = np.concatenate([r['diff_u'] for r in ts_results])
    all_dv = np.concatenate([r['diff_v'] for r in ts_results])
    n_ts   = len(ts_results)
    n_pts  = len(all_du)

    # MLE Gaussian fit
    mu_u, sig_u = stats.norm.fit(all_du)
    mu_v, sig_v = stats.norm.fit(all_dv)

    # Normality test: Shapiro-Wilk up to 5000 samples, D'Agostino-Pearson above
    if n_pts <= 5000:
        test_name = "Shapiro-Wilk"
        _, p_u = stats.shapiro(all_du)
        _, p_v = stats.shapiro(all_dv)
    else:
        test_name = "D'Agostino-Pearson"
        _, p_u = stats.normaltest(all_du)
        _, p_v = stats.normaltest(all_dv)

    result_u = "Gaussiana"     if p_u > 0.05 else "Não-Gaussiana"
    result_v = "Gaussiana"     if p_v > 0.05 else "Não-Gaussiana"

    print("\n" + "=" * 60)
    print("HISTOGRAMA AGREGADO (TODOS OS TIMESTAMPS):")
    print("=" * 60)
    print(f"  Timestamps incluídos: {n_ts}")
    print(f"  Total de pontos:      {n_pts}")
    print(f"  Teste de normalidade: {test_name}")
    print(f"\n  U | μ={mu_u:+.6f}  σ={sig_u:.6f}  p={p_u:.10f}  [{result_u}]")
    print(f"  V | μ={mu_v:+.6f}  σ={sig_v:.6f}  p={p_v:.10f}  [{result_v}]")
    print("=" * 60)

    # Bins using Sturges' rule on the combined sample
    n_bins = int(np.ceil(np.log2(n_pts) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'Distribuição Agregada das Diferenças (NRT − MY) – {n_ts} timestamps, {n_pts} pontos',
        fontsize=14, weight='bold'
    )

    for ax, (comp, data, mu, sig, p_val, color) in zip(axes, [
        ('U', all_du, mu_u, sig_u, p_u, 'steelblue'),
        ('V', all_dv, mu_v, sig_v, p_v, 'coral'),
    ]):
        ax.hist(data, bins=n_bins, density=True,
                color=color, alpha=0.7, edgecolor='black', linewidth=0.4,
                label=f'{n_pts} amostras')

        # Fitted Gaussian curve
        x_fit = np.linspace(data.min(), data.max(), 400)
        ax.plot(x_fit, stats.norm.pdf(x_fit, mu, sig),
                'k-', linewidth=2.0, label=f'N(μ={mu:+.4f}, σ={sig:.4f})')

        ax.axvline(0,  color='black',   linestyle='--', linewidth=0.9, label='zero')
        ax.axvline(mu, color='darkred', linestyle=':',  linewidth=1.2, label=f'μ = {mu:+.4f}')

        result_str = "Gaussiana" if p_val > 0.05 else "Não-Gaussiana"
        color_title = 'darkgreen' if p_val > 0.05 else 'darkred'
        ax.set_title(
            f'Componente {comp}  |  {test_name}: p = {p_val:.10f}  [{result_str}]',
            fontsize=11, weight='bold', color=color_title
        )
        ax.set_xlabel('Diferença (m/s)', fontsize=11)
        ax.set_ylabel('Densidade de probabilidade', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_all_timestamps_angle_histogram(ts_results):
    # Aggregates vector-by-vector angular variation (NRT vs MY) from all timestamps
    # and builds a histogram with circular statistics for analysis.
    # Args: ts_results (list of dicts returned by analyze_all_timestamps_distribution)
    # Returns: fig
    if not ts_results:
        print("Nenhum resultado para plotar.")
        return None

    all_dtheta = np.concatenate([
        r['diff_theta_deg'] for r in ts_results
        if 'diff_theta_deg' in r and len(r['diff_theta_deg']) > 0
    ])
    all_dtheta_time = np.concatenate([
        r['diff_theta_time_str'] for r in ts_results
        if 'diff_theta_time_str' in r and len(r['diff_theta_time_str']) > 0
    ])

    if len(all_dtheta) == 0:
        print("Nenhuma variação angular disponível para plotar.")
        return None

    n_ts = len(ts_results)
    n_pts = len(all_dtheta)

    # Circular stats (angles in radians)
    all_rad = np.radians(all_dtheta)
    mean_circ_rad = stats.circmean(all_rad, high=np.pi, low=-np.pi)
    std_circ_rad = stats.circstd(all_rad, high=np.pi, low=-np.pi)

    mean_circ_deg = float(np.degrees(mean_circ_rad))
    std_circ_deg = float(np.degrees(std_circ_rad))

    # Complementary linear stats for practical interpretation
    mean_lin = float(np.mean(all_dtheta))
    std_lin = float(np.std(all_dtheta))
    q25 = float(np.percentile(all_dtheta, 25))
    q05 = float(np.percentile(all_dtheta, 5))
    q50 = float(np.percentile(all_dtheta, 50))
    q75 = float(np.percentile(all_dtheta, 75))
    q95 = float(np.percentile(all_dtheta, 95))

    # Outliers by IQR rule
    iqr = q75 - q25
    outlier_low = q25 - 1.5 * iqr
    outlier_high = q75 + 1.5 * iqr
    outlier_mask = (all_dtheta < outlier_low) | (all_dtheta > outlier_high)
    outliers = all_dtheta[outlier_mask]
    outlier_times = all_dtheta_time[outlier_mask]
    n_outliers = len(outliers)
    pct_outliers = (100.0 * n_outliers / n_pts) if n_pts > 0 else 0.0

    # Shapiro-Wilk test for normality
    _, p_sw_theta = stats.shapiro(all_dtheta)
    test_name = "Shapiro-Wilk"
    is_gaussian = "GAUSSIANA" if p_sw_theta > 0.05 else "NÃO-GAUSSIANA"

    outlier_time_counts = {}
    for ts in outlier_times:
        outlier_time_counts[ts] = outlier_time_counts.get(ts, 0) + 1
    outlier_time_sorted = sorted(outlier_time_counts.items(), key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 60)
    print("HISTOGRAMA AGREGADO DA VARIAÇÃO ANGULAR (NRT vs MY):")
    print("=" * 60)
    print(f"  Timestamps incluídos: {n_ts}")
    print(f"  Total de pontos:      {n_pts}")
    print(f"  Média circular:       {mean_circ_deg:+.4f}°")
    print(f"  Desvio circular:      {std_circ_deg:.4f}°")
    print(f"  Média linear:         {mean_lin:+.4f}°")
    print(f"  Desvio linear:        {std_lin:.4f}°")
    print(f"  Percentis (5/50/95):  {q05:+.4f}° / {q50:+.4f}° / {q95:+.4f}°")
    print(f"  Outliers (IQR):       {n_outliers} ({pct_outliers:.2f}%)")
    print(f"  Limites IQR:          [{outlier_low:+.4f}°, {outlier_high:+.4f}°]")
    print(f"  Teste {test_name}:     p-value = {p_sw_theta:.10f} [{is_gaussian}]")
    if n_outliers > 0:
        print("  Timestamps com outliers (top 10):")
        for ts, count in outlier_time_sorted[:10]:
            print(f"    {ts}: {count} pontos outliers")
    print("=" * 60)

    # 5-degree bins in [-180, 180]
    bins = np.arange(-180, 185, 5)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5.2))
    fig.suptitle(
        f'Distribuição Agregada da Variação Angular (NRT − MY) – {n_ts} timestamps, {n_pts} pontos',
        fontsize=14, weight='bold'
    )

    ax.hist(all_dtheta, bins=bins, density=True,
            color='mediumpurple', alpha=0.75, edgecolor='black', linewidth=0.4,
            label=f'{n_pts} amostras')

    ax.axvline(0.0, color='black', linestyle='--', linewidth=1.0, label='0° (mesma direção)')
    ax.axvline(mean_circ_deg, color='darkred', linestyle=':', linewidth=1.3,
               label=f'Média circular = {mean_circ_deg:+.2f}°')
    ax.axvline(outlier_low, color='orange', linestyle='--', linewidth=1.1,
               label=f'Limite outlier inf. = {outlier_low:+.2f}°')
    ax.axvline(outlier_high, color='orange', linestyle='--', linewidth=1.1,
               label=f'Limite outlier sup. = {outlier_high:+.2f}°')

    # Rug marks for outliers at the bottom of plot
    if n_outliers > 0:
        y_rug = np.full(n_outliers, 0.0)
        ax.plot(outliers, y_rug, '|', color='orange', markersize=7,
                markeredgewidth=1.1, alpha=0.85, label=f'Outliers = {n_outliers}')

    color_title = 'darkgreen' if p_sw_theta > 0.05 else 'darkred'
    ax.set_title(
        f'Variação Angular Δθ (vetor a vetor)  |  {test_name}: p = {p_sw_theta:.10f}  [{is_gaussian}]',
        fontsize=11, weight='bold', color=color_title
    )
    ax.set_xlabel('Variação angular Δθ (graus)', fontsize=11)
    ax.set_ylabel('Densidade de probabilidade', fontsize=11)
    ax.set_xlim(-180, 180)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_all_timestamps_magnitude_histogram(ts_results):
    # Aggregates vector-by-vector magnitude variation (NRT vs MY) from all timestamps
    # and builds a histogram for analysis.
    # Args: ts_results (list of dicts returned by analyze_all_timestamps_distribution)
    # Returns: fig
    if not ts_results:
        print("Nenhum resultado para plotar.")
        return None

    all_dmag = np.concatenate([
        r['diff_mag'] for r in ts_results
        if 'diff_mag' in r and len(r['diff_mag']) > 0
    ])

    if len(all_dmag) == 0:
        print("Nenhuma variação de magnitude disponível para plotar.")
        return None

    n_ts = len(ts_results)
    n_pts = len(all_dmag)

    mean_mag = float(np.mean(all_dmag))
    std_mag = float(np.std(all_dmag))
    q05 = float(np.percentile(all_dmag, 5))
    q50 = float(np.percentile(all_dmag, 50))
    q95 = float(np.percentile(all_dmag, 95))

    mu_fit, sig_fit = stats.norm.fit(all_dmag)

    # Shapiro-Wilk test for normality
    _, p_mag = stats.shapiro(all_dmag)
    test_name = "Shapiro-Wilk"
    result_mag = "Gaussiana" if p_mag > 0.05 else "Não-Gaussiana"

    print("\n" + "=" * 60)
    print("HISTOGRAMA AGREGADO DA VARIAÇÃO DE MAGNITUDE (NRT vs MY):")
    print("=" * 60)
    print(f"  Timestamps incluídos: {n_ts}")
    print(f"  Total de pontos:      {n_pts}")
    print(f"  Média:               {mean_mag:+.6f} m/s")
    print(f"  Desvio padrão:        {std_mag:.6f} m/s")
    print(f"  Percentis (5/50/95):  {q05:+.6f} / {q50:+.6f} / {q95:+.6f} m/s")
    print(f"  Teste {test_name}:     p-value = {p_mag:.10f} [{result_mag}]")
    print("=" * 60)

    n_bins = int(np.ceil(np.log2(n_pts) + 1))

    fig, ax = plt.subplots(1, 1, figsize=(12, 5.2))
    fig.suptitle(
        f'Distribuição Agregada da Variação de Magnitude (NRT − MY) – {n_ts} timestamps, {n_pts} pontos',
        fontsize=14, weight='bold'
    )

    ax.hist(all_dmag, bins=n_bins, density=True,
            color='teal', alpha=0.75, edgecolor='black', linewidth=0.4,
            label=f'{n_pts} amostras')

    x_fit = np.linspace(all_dmag.min(), all_dmag.max(), 400)
    ax.plot(x_fit, stats.norm.pdf(x_fit, mu_fit, sig_fit),
            'k-', linewidth=1.8, label=f'N(μ={mu_fit:+.4f}, σ={sig_fit:.4f})')

    ax.axvline(0.0, color='black', linestyle='--', linewidth=1.0, label='0 (mesma magnitude)')
    ax.axvline(mean_mag, color='darkred', linestyle=':', linewidth=1.3,
               label=f'Média = {mean_mag:+.4f} m/s')

    color_title = 'darkgreen' if p_mag > 0.05 else 'darkred'
    ax.set_title(
        f'Variação de Magnitude Δ|V| (vetor a vetor)  |  {test_name}: p = {p_mag:.10f}  [{result_mag}]',
        fontsize=11, weight='bold', color=color_title
    )
    ax.set_xlabel('Variação de magnitude Δ|V| (m/s)', fontsize=11)
    ax.set_ylabel('Densidade de probabilidade', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_qq_all_timestamps(ts_results):
    # Generates Q-Q plots from ALL timestamps aggregated into a single distribution
    # Checks if the combined distribution follows a Gaussian.
    # Uses Shapiro-Wilk for n <= 5000, D'Agostino-Pearson (normaltest) for larger samples.
    # Args: ts_results (list of dicts returned by analyze_all_timestamps_distribution)
    # Returns: fig
    if not ts_results:
        print("Nenhum resultado para plotar.")
        return None

    # Concatenate all timestamps into single arrays
    all_du = np.concatenate([r['diff_u'] for r in ts_results])
    all_dv = np.concatenate([r['diff_v'] for r in ts_results])
    n_ts   = len(ts_results)
    n_pts  = len(all_du)

    # MLE Gaussian fit
    mu_u, sig_u = stats.norm.fit(all_du)
    mu_v, sig_v = stats.norm.fit(all_dv)

    # Normality test: Shapiro-Wilk up to 5000 samples, D'Agostino-Pearson above
    if n_pts <= 5000:
        test_name = "Shapiro-Wilk"
        _, p_u = stats.shapiro(all_du)
        _, p_v = stats.shapiro(all_dv)
    else:
        test_name = "D'Agostino-Pearson"
        _, p_u = stats.normaltest(all_du)
        _, p_v = stats.normaltest(all_dv)

    result_u = "Gaussiana"     if p_u > 0.05 else "Não-Gaussiana"
    result_v = "Gaussiana"     if p_v > 0.05 else "Não-Gaussiana"

    print("\n" + "=" * 60)
    print("Q-Q PLOT AGREGADO (TODOS OS TIMESTAMPS):")
    print("=" * 60)
    print(f"  Timestamps incluídos: {n_ts}")
    print(f"  Total de pontos:      {n_pts}")
    print(f"  Teste de normalidade: {test_name}")
    print(f"\n  U | μ={mu_u:+.6f}  σ={sig_u:.6f}  p={p_u:.10f}  [{result_u}]")
    print(f"  V | μ={mu_v:+.6f}  σ={sig_v:.6f}  p={p_v:.10f}  [{result_v}]")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f'Q-Q Plot Agregado (NRT − MY) – {n_ts} timestamps, {n_pts} pontos',
        fontsize=14, weight='bold'
    )

    for ax, (comp, data, mu, sig, p_val) in zip(axes, [
        ('U', all_du, mu_u, sig_u, p_u),
        ('V', all_dv, mu_v, sig_v, p_v),
    ]):
        stats.probplot(data, dist="norm", plot=ax)
        result_str = "Gaussiana" if p_val > 0.05 else "Não-Gaussiana"
        color_title = 'darkgreen' if p_val > 0.05 else 'darkred'
        ax.set_title(
            f'Componente {comp}  |  {test_name}: p = {p_val:.10f}  [{result_str}]',
            fontsize=11, weight='bold', color=color_title
        )
        ax.set_xlabel('Quantis Teóricos', fontsize=11)
        ax.set_ylabel('Quantis Observados', fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def get_data_path(filename):
    # Builds a path relative to the script directory
    # Args: filename (str) - name of the file in the analysis directory
    # Returns: absolute path to the file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, filename)
    return data_path


def resolve_dataset_path(filename):
    # Resolves dataset file path with priority to analysis/modelos
    # Args: filename (str) - dataset file name, relative path, or absolute path
    # Returns: absolute path to an existing file
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.path.join(script_dir, "modelos", filename),
        os.path.join(script_dir, filename),
        os.path.join(script_dir, "BaixadoDia4", filename),
        os.path.join(script_dir, "BaixadoDia5", filename),
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            return path

    searched_paths = "\n".join(f"  - {p}" for p in candidate_paths)
    raise FileNotFoundError(
        f"Arquivo não encontrado: {filename}\n"
        f"Locais buscados:\n{searched_paths}"
    )

def print_information(nrt_dataset, metrics):
    # Prints dataset information and calculated metrics
    # Args: nrt_dataset (original NRT dataset), metrics (dict with calculated metrics)
    

    print("=" * 60)
    print("INFORMAÇÕES DO ARQUIVO:")
    print("=" * 60)
    print("\nDIMENSÕES:")
    print(nrt_dataset.sizes)
    print("\nVARIÁVEIS:")
    for var in nrt_dataset.data_vars:
        print(f"  - {var}: {nrt_dataset[var].dims} | Shape: {nrt_dataset[var].shape}")
    print("\nCOORDENADAS:")
    for coord in nrt_dataset.coords:
        print(f"  - {coord}: {nrt_dataset[coord].shape}")
    
    print("\n" + "=" * 60)
    print("MÉTRICAS DE ERRO (NRT - MY):")
    print("=" * 60)
    
    print("\n1. RMSE por componente:")
    print(f"  - RMSE u: {metrics['rmse_u']:.6f} m/s")
    print(f"  - RMSE v: {metrics['rmse_v']:.6f} m/s")
    
    print("\n2. Erro Vetorial:")
    print(f"  - RMSE vetorial: {metrics['rmse_vector']:.6f} m/s")
    print(f"  - Erro médio vetorial: {metrics['mean_vector_error']:.6f} m/s")
    print(f"  - Erro máximo vetorial: {metrics['max_vector_error']:.6f} m/s")

def main():
    # Main function - executes complete analysis
    
    # Input parameters
    file_ref = resolve_dataset_path('corrente_cmems_mod_glo_phy_anfc_0.083deg_RUN_2025-06-12.nc')
    file_cmp = resolve_dataset_path('corrente_remo-hycom124v2_2d_map-runs-remo_hycom124v2_2d_map_RUN_2025-06-12.nc')
    label_ref = "ANFC"
    label_cmp = "REMO Hycom"
    datetime_str = "2026-06-12T00:00:00"
    lat_min_req, lat_max_req = None, None
    lon_min_req, lon_max_req = None, None
    n_expand = 3

    # GIF temporal (modelo de comparação fixo)
    enable_temporal_gif = False
    fixed_cmp_time_str = datetime_str
    gif_start_time_str = "2025-06-12T00:00:00"
    gif_end_time_str = "2025-06-16T00:00:00"
    gif_fps = 30
    gif_frame_stride = 1
    gif_max_frames = 96
    gif_output_path = get_data_path(os.path.join('gifs', 'evolucao_temporal_ref_variando_cmp_fixo.gif'))

    # Estatísticas agregadas (mais custosas) - desative para focar no GIF
    run_full_aggregated_analysis = True
    
    # 1. Load data
    print(f"  Arquivo NRT: {file_ref}")
    print(f"  Arquivo MY:  {file_cmp}")
    nrt, my, nrt_slice, my_slice, actual_time_nrt, actual_time_my = load_data(file_ref, file_cmp, datetime_str)
    
    # Display available timesteps
    print("\n" + "=" * 60)
    print("TIMESTEPS DISPONÍVEIS NO ARQUIVO NRT:")
    print("=" * 60)
    print(f"Total de timesteps: {len(nrt.time.values)}")
    print("\nLista de datas/horas:")
    for i, time in enumerate(nrt.time.values):
        time_str = str(time).replace('T', ' ').split('.')[0]  # Format: YYYY-MM-DD HH:MM:SS
        print(f"  {i+1:3d}. {time_str}")
    print("=" * 60)
    print(f"\nTimestep selecionado para análise: {datetime_str.replace('T', ' ')}\n")
    
    # 2. Extract components
    u, v, lon, lat = extract_components(nrt_slice)
    u_my, v_my, lon_my, lat_my = extract_components(my_slice)
    
    # 3. Calculate spatial crop
    if all(v is None for v in (lat_min_req, lat_max_req, lon_min_req, lon_max_req)):
        print("\nIntervalo solicitado: grade completa")
    else:
        lat_req_str = f"[{lat_min_req}, {lat_max_req}]" if (lat_min_req is not None or lat_max_req is not None) else "completo"
        lon_req_str = f"[{lon_min_req}, {lon_max_req}]" if (lon_min_req is not None or lon_max_req is not None) else "completo"
        print(f"\nIntervalo solicitado: lat {lat_req_str}, lon {lon_req_str}")
    lat_indices_nrt, lon_indices_nrt, lat_min, lat_max, lon_min, lon_max = calculate_spatial_crop(
        lat, lon, lat_min_req, lat_max_req, lon_min_req, lon_max_req, n_expand
    )

    lat_indices_my, lon_indices_my, _, _, _, _ = calculate_spatial_crop(
        lat_my, lon_my, lat_min, lat_max, lon_min, lon_max, 0
    )
    print(f"Intervalo ampliado: lat [{lat_min:.4f}, {lat_max:.4f}], "
          f"lon [{lon_min:.4f}, {lon_max:.4f}]")
    print(f"Pontos encontrados NRT: {len(lat_indices_nrt)} lat x {len(lon_indices_nrt)} lon")
    print(f"Pontos encontrados MY:  {len(lat_indices_my)} lat x {len(lon_indices_my)} lon")
    
    # 4. Apply crop
    if len(lat_indices_nrt) > 0 and len(lon_indices_nrt) > 0 and len(lat_indices_my) > 0 and len(lon_indices_my) > 0:
        u, v, lat, lon = apply_crop(u, v, lat, lon, lat_indices_nrt, lon_indices_nrt)
        u_my, v_my, lat_my, lon_my = apply_crop(u_my, v_my, lat_my, lon_my,
                                                lat_indices_my, lon_indices_my)
    else:
        print("AVISO: Nenhum ponto encontrado no recorte!")
        return
    
    # 5. Ensure correct dimensions
    u, v = ensure_dimensions(u, v)
    u_my, v_my = ensure_dimensions(u_my, v_my)
    
    # 6. Align grids
    u_my_aligned, v_my_aligned = align_grids(u_my, v_my, lon_my, lat_my, lon, lat)
    
    # 7. Calculate metrics
    metrics = calculate_metrics(u, v, u_my_aligned, v_my_aligned)
    
    # 8. Plot
    fig, axes = plot_comparison(u, v, u_my_aligned, v_my_aligned,
                                   metrics['diff_u'], metrics['diff_v'],
                                   lon, lat, lon_min, lon_max, lat_min, lat_max,
                                   datetime_str,
                                   label_ref=label_ref, label_cmp=label_cmp)
    
    # 9. Print information
    print_information(nrt, metrics)
    
    # 9.5. Analyze error distribution
    #analyze_error_distribution(metrics['diff_u'], metrics['diff_v'])

    # 9.6. Q-Q plot do timestamp selecionado
    ##fig_qq_selected = analyze_gaussian_fit(metrics['diff_u'], metrics['diff_v'])

    # 10. Plot histograms
    #fig_hist = plot_histogramdif(metrics['diff_u'], metrics['diff_v'], u_my_aligned, v_my_aligned)

    ts_results = None
    if run_full_aggregated_analysis:
        # 11. Analyse distribution over all timestamps
        ts_results = analyze_all_timestamps_distribution(
            nrt, my,
            lat_indices_nrt, lon_indices_nrt,
            lat_indices_my, lon_indices_my
        )

        # 12. Plot aggregated histogram across all timestamps with Gaussian fit
        fig_agg_hist = plot_all_timestamps_histogram(ts_results)

        # 12.1. Plot aggregated histogram of vector-by-vector angular variation
        fig_agg_angle = plot_all_timestamps_angle_histogram(ts_results)

        # 12.2. Plot aggregated histogram of vector-by-vector magnitude variation
        fig_agg_magnitude = plot_all_timestamps_magnitude_histogram(ts_results)

    # 12.3. GIF de evolução temporal (ref variando, cmp fixo)
    if enable_temporal_gif:
        generate_timestamp_evolution_gif(
            nrt, my,
            lat_indices_nrt, lon_indices_nrt,
            lat_indices_my, lon_indices_my,
            lon_min, lon_max, lat_min, lat_max,
            label_ref, label_cmp,
            fixed_cmp_time_str=fixed_cmp_time_str,
            gif_start_time_str=gif_start_time_str,
            gif_end_time_str=gif_end_time_str,
            output_gif_path=gif_output_path,
            fps=gif_fps,
            frame_stride=gif_frame_stride,
            max_frames=gif_max_frames
        )

    # 13. Q-Q plot aggregated across all timestamps
    ##fig_qq = plot_qq_all_timestamps(ts_results)

    plt.show()

if __name__ == "__main__":
    main()
