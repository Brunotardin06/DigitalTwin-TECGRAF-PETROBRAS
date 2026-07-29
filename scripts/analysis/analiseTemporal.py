import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
from datetime import datetime

# Importa funções do analiseVetorial
from analiseVetorial import (
    extract_components,
    align_grids,
)

def calculate_metrics_single_point(u_ref, v_ref, u_cmp, v_cmp):
    """
    Calcula métricas para um ponto ÚNICO e INSTANTÂNEO (sem agregação espacial).
    Simplesmente faz a diferença entre valores escalares.
    Retorna: dict com erros instantâneos
    """
    # Diferenças simples (sem média)
    diff_u = float(u_ref) - float(u_cmp)
    diff_v = float(v_ref) - float(v_cmp)
    
    # Raiz dos erros ao quadrado (√(diff²) - explicitamente "raiz do erro quadrático")
    error_u = float(np.sqrt(diff_u**2))
    error_v = float(np.sqrt(diff_v**2))
    
    # Magnitude dos vetores
    mag_ref = float(np.sqrt(float(u_ref)**2 + float(v_ref)**2))
    mag_cmp = float(np.sqrt(float(u_cmp)**2 + float(v_cmp)**2))
    
    # Magnitude do vetor diferença (erro de magnitude)
    magnitude_error = float(np.sqrt(diff_u**2 + diff_v**2))
    
    # Erro relativo (%)
    relative_error_pct = (magnitude_error / mag_ref * 100) if mag_ref > 0 else 0.0
    
    return {
        'u_ref': float(u_ref),
        'v_ref': float(v_ref),
        'u_cmp': float(u_cmp),
        'v_cmp': float(v_cmp),
        'diff_u': diff_u,
        'diff_v': diff_v,
        'error_u': error_u,
        'error_v': error_v,
        'mag_ref': mag_ref,
        'mag_cmp': mag_cmp,
        'magnitude_error': magnitude_error,
        'relative_error_pct': relative_error_pct,
    }


def calculate_metrics(u_nrt, v_nrt, u_my, v_my):
    """
    Calcula métricas para GRID COMPLETO (com agregação espacial via nanmean).
    Usa a função original do analiseVetorial que já está importada.
    """
    # Detecta se é um ponto único (escalar) ou grid completo
    try:
        if hasattr(u_nrt, 'ndim'):
            if u_nrt.ndim == 0:
                # É um escalar - usa cálculo instantâneo
                return calculate_metrics_single_point(u_nrt, v_nrt, u_my, v_my)
    except:
        pass
    
    # Para grid completo, usa lógica original
    from analiseVetorial import calculate_metrics as calc_metrics_grid
    return calc_metrics_grid(u_nrt, v_nrt, u_my, v_my)


def check_resolution(lon_ref, lat_ref, lon_cmp, lat_cmp):
    """
    Verifica as resoluções dos grids e retorna informação sobre qual é mais grossa.
    Se os dados são de ponto único (escalares), retorna (0, 0, 0, 0, False).
    Retorna: (ref_spacing_lon, ref_spacing_lat, cmp_spacing_lon, cmp_spacing_lat, cmp_is_finer)
    """
    # Se lon/lat são escalares (ponto único), skip da verificação
    try:
        if lon_ref.ndim == 0 or lat_ref.ndim == 0 or lon_cmp.ndim == 0 or lat_cmp.ndim == 0:
            return 0, 0, 0, 0, False
    except:
        return 0, 0, 0, 0, False
    
    try:
        ref_diff_lon = np.diff(lon_ref.values)
        ref_diff_lat = np.diff(lat_ref.values)
        cmp_diff_lon = np.diff(lon_cmp.values)
        cmp_diff_lat = np.diff(lat_cmp.values)
        
        if len(ref_diff_lon) == 0 or len(ref_diff_lat) == 0 or len(cmp_diff_lon) == 0 or len(cmp_diff_lat) == 0:
            return 0, 0, 0, 0, False
    except:
        return 0, 0, 0, 0, False
    
    lon_spacing_ref = np.abs(ref_diff_lon).mean()
    lat_spacing_ref = np.abs(ref_diff_lat).mean()
    
    lon_spacing_cmp = np.abs(cmp_diff_lon).mean()
    lat_spacing_cmp = np.abs(cmp_diff_lat).mean()
    
    # True se comparison é mais fino (menor espaçamento)
    cmp_is_finer = (lon_spacing_cmp < lon_spacing_ref) or (lat_spacing_cmp < lat_spacing_ref)
    
    return lon_spacing_ref, lat_spacing_ref, lon_spacing_cmp, lat_spacing_cmp, cmp_is_finer


def degrade_grid(u_cmp, v_cmp, lon_cmp, lat_cmp, lon_ref, lat_ref):
    """
    Degrada o grid de comparison para o grid de reference (se comparison for mais fino).
    Usa agregação (coarsen + mean) sem interpolação.
    Para dados escalares (ponto único), retorna sem degradação.
    Retorna: (u_degraded, v_degraded, lon_degraded, lat_degraded)
    """
    # Se dados são escalares, retorna como está
    try:
        if u_cmp.ndim == 0 or lon_cmp.ndim == 0 or lat_cmp.ndim == 0:
            return u_cmp, v_cmp, lon_cmp, lat_cmp
    except:
        return u_cmp, v_cmp, lon_cmp, lat_cmp
    
    try:
        same_lon = np.array_equal(lon_ref.values, lon_cmp.values)
        same_lat = np.array_equal(lat_ref.values, lat_cmp.values)
    except:
        # Se não conseguir comparar, retorna original
        return u_cmp, v_cmp, lon_cmp, lat_cmp
    
    if same_lon and same_lat:
        # Grids já estão alinhados
        return u_cmp, v_cmp, lon_cmp, lat_cmp
    
    try:
        # Calcula espaçamentos
        ref_diff_lon = np.diff(lon_ref.values)
        ref_diff_lat = np.diff(lat_ref.values)
        cmp_diff_lon = np.diff(lon_cmp.values)
        cmp_diff_lat = np.diff(lat_cmp.values)
        
        if len(ref_diff_lon) == 0 or len(cmp_diff_lon) == 0:
            return u_cmp, v_cmp, lon_cmp, lat_cmp
        
        lon_spacing_ref = np.abs(ref_diff_lon).mean()
        lat_spacing_ref = np.abs(ref_diff_lat).mean()
        lon_spacing_cmp = np.abs(cmp_diff_lon).mean()
        lat_spacing_cmp = np.abs(cmp_diff_lat).mean()
    except:
        return u_cmp, v_cmp, lon_cmp, lat_cmp
    
    # Se comparison é mais fino, degrada
    if lon_spacing_cmp < lon_spacing_ref or lat_spacing_cmp < lat_spacing_ref:
        try:
            # Calcula fatores de degradação
            lon_factor = int(np.round(lon_spacing_ref / lon_spacing_cmp))
            lat_factor = int(np.round(lat_spacing_ref / lat_spacing_cmp))
            
            lon_factor = max(1, lon_factor)
            lat_factor = max(1, lat_factor)
            
            if lon_factor > 1 or lat_factor > 1:
                # Infere dimensões
                lat_dims = [d for d in u_cmp.dims if 'lat' in d.lower()]
                lon_dims = [d for d in u_cmp.dims if 'lon' in d.lower()]
                
                if lat_dims and lon_dims:
                    lat_dim = lat_dims[0]
                    lon_dim = lon_dims[0]
                    
                    print(f"  Degradacao: lon_factor={lon_factor}, lat_factor={lat_factor}")
                    u_degraded = u_cmp.coarsen({lon_dim: lon_factor, lat_dim: lat_factor}, boundary='trim').mean()
                    v_degraded = v_cmp.coarsen({lon_dim: lon_factor, lat_dim: lat_factor}, boundary='trim').mean()
                    
                    # Extrai coordenadas degradadas
                    lon_degraded = u_degraded[lon_dim]
                    lat_degraded = u_degraded[lat_dim]
                    
                    return u_degraded, v_degraded, lon_degraded, lat_degraded
        except Exception as e:
            print(f"  Erro durante degradacao: {e}")
            return u_cmp, v_cmp, lon_cmp, lat_cmp
    
    # Se não precisa degradar, retorna original
    return u_cmp, v_cmp, lon_cmp, lat_cmp


def loadData(fileReference, fileComparison1, fileComparison2, timeStepTarget, timeStepStart, timeStepEnd, latitude=None, longitude=None, passoHoras=1):
    """
    Carrega dados de referência e dois arquivos de comparação (24h cada = 48h total).
    Fixa um timestep da reference e itera sobre todos os timesteps de comparison.
    Se latitude/longitude são fornecidos, extrai apenas o ponto mais próximo.
    Retorna lista de resultados com métricas temporais.
    """
    reference = xr.open_dataset(fileReference)
    
    # Abre e concatena os dois arquivos de comparação
    comparison1 = xr.open_dataset(fileComparison1)
    comparison2 = xr.open_dataset(fileComparison2)
    
    # Concatena os dois datasets ao longo da dimensão tempo
    comparison = xr.concat([comparison1, comparison2], dim='time')
    
    # Remove duplicatas de tempo e ordena
    comparison = comparison.drop_duplicates(dim='time').sortby('time')
    
    print(f"Arquivo Comparison 1: {len(comparison1.time)} timestamps")
    print(f"Arquivo Comparison 2: {len(comparison2.time)} timestamps")
    print(f"Comparison concatenado: {len(comparison.time)} timestamps\n")

    # Seleciona o timestep fixo da referência
    referenceFixed = reference.sel(time=timeStepTarget, method="nearest")
    referenceFixedTimeStr = str(referenceFixed.time.values).replace('T', ' ').split('.')[0]
    
    print(f"Referência fixa em: {referenceFixedTimeStr}\n")
    
    # Se latitude/longitude são fornecidas, extrai o ponto mais próximo
    if latitude is not None and longitude is not None:
        print(f"{'='*60}")
        print(f"MODO: ANALISE DE PONTO UNICO")
        print(f"{'='*60}")
        print(f"Ponto solicitado: Lat={latitude}, Lon={longitude}")
        
        referenceFixed = referenceFixed.sel(latitude=latitude, longitude=longitude, method="nearest")
        comparison = comparison.sel(latitude=latitude, longitude=longitude, method="nearest")
        
        actual_lat = float(referenceFixed.latitude.values) if hasattr(referenceFixed, 'latitude') else latitude
        actual_lon = float(referenceFixed.longitude.values) if hasattr(referenceFixed, 'longitude') else longitude
        print(f"Ponto encontrado:  Lat={actual_lat}, Lon={actual_lon}")
        if abs(actual_lat - latitude) > 0.05 or abs(actual_lon - longitude) > 0.05:
            print(f"[AVISO] Diferenca > 0.05 em relacao ao solicitado")
        print(f"{'='*60}\n")
    else:
        print(f"{'='*60}")
        print(f"MODO: ANALISE DE GRID COMPLETO")
        print(f"{'='*60}")
        print(f"Todos os {referenceFixed.latitude.size * referenceFixed.longitude.size} pontos serao analisados")
        print(f"{'='*60}\n")
    
    # Define intervalo de comparison para iterar
    startTime = comparison.sel(time=timeStepStart, method="nearest").time.values
    endTime = comparison.sel(time=timeStepEnd, method="nearest").time.values
    
    # Slice sem method
    comparisonWindow = comparison.sel(time=slice(startTime, endTime))
    comparisonTimes = comparisonWindow.time.values

    if len(comparisonTimes) == 0:
        print("ERRO: Nenhum timestamp encontrado em comparison")
        return None
    
    resultados = []
    
    # Extrai componentes da referência fixa UMA VEZ
    uRef, vRef, lonRef, latRef = extract_components(referenceFixed)
    
    # Verifica resolução no primeiro timestep de comparison
    first_check_done = False

    for i, tempoComp in enumerate(comparisonTimes):
        tempoCompStr = str(tempoComp).replace('T', ' ').split('.')[0]

        try:
            comparisonT = comparison.sel(time=tempoComp, method='nearest')
            
            # Usa extract_components do analiseVetorial
            uCmp, vCmp, lonCmp, latCmp = extract_components(comparisonT)
            
            # Verifica e informa sobre resoluções (apenas na primeira iteração)
            if not first_check_done:
                lon_sp_ref, lat_sp_ref, lon_sp_cmp, lat_sp_cmp, cmp_is_finer = check_resolution(lonRef, latRef, lonCmp, latCmp)
                print(f"Resolucoes detectadas:")
                print(f"  Reference: dlon={lon_sp_ref:.6f}, dlat={lat_sp_ref:.6f}")
                print(f"  Comparison: dlon={lon_sp_cmp:.6f}, dlat={lat_sp_cmp:.6f}")
                if cmp_is_finer:
                    print(f"  -> Comparison eh mais FINO (sera degradado)")
                else:
                    print(f"  -> Grids tem resolucoes similares ou Ref eh mais fino")
                print()
                first_check_done = True
            
            # Degrada grid de comparison se necessário (sem interpolação!)
            uCmpDegraded, vCmpDegraded, lonCmpDegraded, latCmpDegraded = degrade_grid(
                uCmp, vCmp, lonCmp, latCmp, lonRef, latRef
            )
            
            # Usa calculate_metrics do analiseVetorial (ou instantâneo para ponto único)
            metrics = calculate_metrics(uRef, vRef, uCmpDegraded, vCmpDegraded)
            
            resultados.append({
                'tempo': tempoComp,
                'tempoStr': tempoCompStr,
                'errorU': metrics.get('error_u', metrics.get('rmse_u', 0)),  # Erro instantâneo componente U
                'errorV': metrics.get('error_v', metrics.get('rmse_v', 0)),  # Erro instantâneo componente V
                'magnitude_error': metrics.get('magnitude_error', metrics.get('rmse_vector', 0)),  # Erro de magnitude
                'relativeErrorPct': metrics['relative_error_pct'],
                'uRef': metrics.get('u_ref'),
                'vRef': metrics.get('v_ref'),
                'uCmp': metrics.get('u_cmp'),
                'vCmp': metrics.get('v_cmp'),
                'diffU': metrics.get('diff_u'),
                'diffV': metrics.get('diff_v'),
            })
            
            # Output detalhado para ponto único
            if latitude is not None and longitude is not None:
                print(f"  [{i+1:3d}/{len(comparisonTimes)}] {tempoCompStr}")
                print(f"    u: ref={metrics.get('u_ref', 0):+.4f} | cmp={metrics.get('u_cmp', 0):+.4f} | diff={metrics.get('diff_u', 0):+.4f}")
                print(f"    v: ref={metrics.get('v_ref', 0):+.4f} | cmp={metrics.get('v_cmp', 0):+.4f} | diff={metrics.get('diff_v', 0):+.4f}")
                print(f"    |magnitude_error|={metrics.get('magnitude_error', 0):.4f} m/s | erro_rel={metrics['relative_error_pct']:.1f}%")
            else:
                print(f"  [{i+1:3d}/{len(comparisonTimes)}] {tempoCompStr} | "
                      f"Magnitude_error={metrics.get('rmse_vector', metrics.get('magnitude_error', 0)):.4f} m/s | "
                      f"Erro_rel={metrics['relative_error_pct']:.1f}%")

        except Exception as e:
            print(f"  [{i+1:3d}/{len(comparisonTimes)}] {tempoCompStr} - ERRO: {e}")
    
    # Resumo
    if resultados:
        error_vec = [r['magnitude_error'] for r in resultados]  # Erro de magnitude
        print("\n" + "="*60)
        print("RESUMO FINAL:")
        print("="*60)
        print(f"Timestamps processados: {len(resultados)}")
        print(f"Erro de magnitude min: {min(error_vec):.4f} m/s")
        print(f"Erro de magnitude med: {np.mean(error_vec):.4f} m/s")
        print(f"Erro de magnitude max: {max(error_vec):.4f} m/s")
        print("="*60)
        
        return resultados
    
    return None


def plot_temporal_results(resultados, referenceFixedTimeStr=None):
    """
    Plota análise temporal dos resultados de ponto único (instantâneo).
    3 subplots: Erro componentes U/V, Erro vetorial instantâneo, Erro relativo.
    Adiciona linhas verticais: amarela (timestep de referência) e roxa (menor erro).
    """
    if not resultados or len(resultados) == 0:
        print("ERRO: Nenhum resultado para plotar")
        return
    
    # Extrai dados (todos são erros instantâneos para análise de ponto único)
    tempos_dt = [np.datetime64(r['tempo']) for r in resultados]
    tempos_str = [r['tempoStr'] for r in resultados]
    error_u = [r['errorU'] for r in resultados]     # Erro instantâneo U
    error_v = [r['errorV'] for r in resultados]     # Erro instantâneo V
    error_vec = [r['magnitude_error'] for r in resultados]  # Erro de magnitude
    erro_rel = [r['relativeErrorPct'] for r in resultados]  # Erro relativo (%)
    
    x = np.arange(len(resultados))
    
    # Encontra o timestep com menor erro de magnitude
    idx_min_erro = np.argmin(error_vec)
    
    # Encontra o índice do timestep de referência (se fornecido)
    idx_ref = None
    if referenceFixedTimeStr:
        for i, ts in enumerate(tempos_str):
            if referenceFixedTimeStr in ts:
                idx_ref = i
                break
    
    # Cria figura com 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Erros instantâneos das componentes U e V
    axes[0].plot(x, error_u, 'o-', label='Erro U', color='red', linewidth=2, markersize=6)
    axes[0].plot(x, error_v, 's-', label='Erro V', color='blue', linewidth=2, markersize=6)
    axes[0].set_ylabel('Erro Instantâneo (m/s)', fontsize=11, weight='bold')
    axes[0].set_title('Evolução Temporal: Erros Instantâneos - Componentes U e V', fontsize=12, weight='bold')
    axes[0].grid(True, alpha=0.3)
    if idx_ref is not None:
        axes[0].axvline(x=idx_ref, color='yellow', linestyle='--', linewidth=2.5, label='Ref. Fixo')
    axes[0].axvline(x=idx_min_erro, color='purple', linestyle='--', linewidth=2.5, label='Menor Erro')
    axes[0].legend(loc='upper left', fontsize=10)
    
    # Plot 2: Erro de magnitude
    axes[1].plot(x, error_vec, 'D-', label='Erro de Magnitude', color='green', linewidth=2, markersize=6)
    axes[1].set_ylabel('Erro de Magnitude (m/s)', fontsize=11, weight='bold')
    axes[1].set_title('Evolução Temporal: Erro de Magnitude da Velocidade', fontsize=12, weight='bold')
    axes[1].grid(True, alpha=0.3)
    if idx_ref is not None:
        axes[1].axvline(x=idx_ref, color='yellow', linestyle='--', linewidth=2.5)
    axes[1].axvline(x=idx_min_erro, color='purple', linestyle='--', linewidth=2.5)
    axes[1].legend(loc='upper left', fontsize=10)
    
    # Plot 3: Erro relativo
    axes[2].plot(x, erro_rel, '^-', label='Erro Relativo', color='darkblue', linewidth=2, markersize=6)
    axes[2].set_ylabel('Erro Relativo (%)', fontsize=11, weight='bold')
    axes[2].set_xlabel('Timestamp', fontsize=11, weight='bold')
    axes[2].set_title('Evolução Temporal: Erro Relativo Percentual', fontsize=12, weight='bold')
    axes[2].grid(True, alpha=0.3)
    if idx_ref is not None:
        axes[2].axvline(x=idx_ref, color='yellow', linestyle='--', linewidth=2.5, label='Ref. Fixo')
    axes[2].axvline(x=idx_min_erro, color='purple', linestyle='--', linewidth=2.5, label='Menor Erro')
    axes[2].legend(loc='upper left', fontsize=10)
    
    # Configura eixo x com timestamps
    step = max(1, len(tempos_str) // 10)  # mostra ~10 labels
    axes[2].set_xticks(x[::step])
    axes[2].set_xticklabels(tempos_str[::step], rotation=45, ha='right', fontsize=9)
    
    plt.suptitle('Análise de Defasagem Temporal - Comparação Copernicus vs Comparação Copernicus (Fixo)',
                 fontsize=14, weight='bold', y=1.00)
    plt.tight_layout()
    
    return fig


def main():
    file_ref = 'analysis/modelos/corrente_cmems_mod_glo_phy_anfc_0.083deg_RUN_2025-06-12.nc'
    file_cmp1 = 'analysis/dataset-uv-rep-daily_20250612T1200Z_P20251114T0000(my).nc'
    file_cmp2 = 'analysis/dataset-uv-rep-daily_20250613T1200Z_P20251114T0000(my).nc'
    datetime_str = "2025-06-13T00:00:00"
    
    # Coordenadas de interesse (para análise em ponto específico)
    # Aceita argumentos: python analiseTemporal.py [latitude] [longitude]
    # Exemplo: python analiseTemporal.py -20.0 -40.0
    # Ou use: python analiseTemporal.py (para grid completo)
    latitude = -15 
    longitude = -35 
    
    if len(sys.argv) >= 3:
        try:
            latitude = float(sys.argv[1])
            longitude = float(sys.argv[2])
            print(f"\n{'='*60}")
            print(f"Argumentos de linha de comando detectados")
            print(f"Lat={latitude}, Lon={longitude}")
            print(f"{'='*60}\n")
        except ValueError:
            print(f"Erro: Argumentos inválidos. Use: python analiseTemporal.py [latitude] [longitude]")
            print(f"Exemplo: python analiseTemporal.py -20.0 -40.0")
            sys.exit(1)
    
    # Executa análise temporal com dois arquivos de comparação (48h total)
    resultados = loadData(file_ref, file_cmp1, file_cmp2, datetime_str, "2025-06-12T00:00:00", "2025-06-14T00:00:00", 
                         latitude=latitude, longitude=longitude)
    
    # Extrai o timestep de referência fixo para passar ao plot
    referenceFixedTimeStr = "2025-06-13 00:00:00"
    
    # Plota resultados
    if resultados:
        fig = plot_temporal_results(resultados, referenceFixedTimeStr)
        
        # Salva figura
        if latitude is not None and longitude is not None:
            output_file = f"analysis/gifs/temporal_analysis_lat{latitude}_lon{longitude}.png"
        else:
            output_file = "analysis/gifs/temporal_analysis_grid_completo.png"
        
        try:
            fig.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"\n{'='*60}")
            print(f"Figura salva em: {output_file}")
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"Aviso: Não foi possível salvar figura: {e}")
        
        plt.show()


if __name__ == "__main__":
    main()








