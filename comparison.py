import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import correlate
from post_processing import find_sensitivity_extremes



def compare_signals_plot(signals: list, t_array: np.ndarray, labels: list, save_dir: str,
                        noise_floor: float, normalize=True):
    """
    Compare signals with proper noise normalization.
    
    Args:
        signals: List of clean signal arrays
        t_array: Time axis in seconds (shape should be (N,) or (1,N))
        labels: Names for each signal
        save_dir: Directory to save plots
        noise_floor: Standard deviation of noise (Pa)
        normalize: Whether to normalize amplitudes
    """
    plt.figure(figsize=(12, 6))
    
    # Ensure time array is properly shaped (convert from (1,N) to (N,))
    t = t_array.flatten() * 1e6  # Convert to μs
    
    for sig, lbl in zip(signals, labels):
        # Ensure signal is properly shaped
        sig = sig.flatten()
        
        # Calculate power SNR (more stable than peak SNR)
        signal_power = np.mean(sig**2)
        snr_db = 10*np.log10(signal_power / (noise_floor**2 + 1e-12))
        
        # Normalize if requested
        if normalize:
            sig = sig / (np.max(np.abs(sig)) + 1e-12)
        
        plt.plot(t, sig, label=f"{lbl} (SNR={snr_db:.1f} dB)")
    
    plt.xlabel("Time (μs)")
    plt.ylabel("Normalized Pressure" if normalize else "Pressure (Pa)")
    plt.title("Signal Comparison")
    plt.legend()
    plt.grid(True)
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "signal_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved signal comparison to: {save_path}")

def calculate_autocorr_metrics(signal: np.ndarray, noise_floor: float):
    """
    Calculate autocorrelation metrics properly.
    Returns:
        - peak_height (σ units)
        - mainlobe_width (samples)
        - peak_to_sidelobe (dB)
    """
    # Normalize signal to unit energy
    h = (signal - np.mean(signal)) / (np.linalg.norm(signal) + 1e-12)
    
    # Compute autocorrelation
    corr = np.correlate(h, h, mode='full')
    corr = corr / (noise_floor + 1e-12)  # Convert to σ units
    
    # Find main peak
    peak_idx = np.argmax(np.abs(corr))
    peak_val = corr[peak_idx]
    
    # Calculate -3dB width
    half_power = peak_val / np.sqrt(2)
    left = peak_idx
    while left > 0 and corr[left] > half_power:
        left -= 1
    right = peak_idx
    while right < len(corr)-1 and corr[right] > half_power:
        right += 1
    width = right - left
    
    # Calculate peak-to-sidelobe ratio
    sidelobe_mask = np.ones(len(corr), dtype=bool)
    guard = width * 2
    sidelobe_mask[max(0, peak_idx-guard):min(len(corr), peak_idx+guard)] = False
    if np.any(sidelobe_mask):
        psr = 20*np.log10(peak_val / (np.max(np.abs(corr[sidelobe_mask])) + 1e-12))
    else:
        psr = np.inf
    
    return peak_val, width, psr

def compare_autocorrelations(signals: list, labels: list, save_dir: str, 
                            noise_floor: float, t_array: np.ndarray):
    """
    Compare autocorrelations with proper noise normalization.
    """
    plt.figure(figsize=(12, 6))
    dt = t_array[1] - t_array[0]
    
    for sig, lbl in zip(signals, labels):
        peak, width, psr = calculate_autocorr_metrics(sig, noise_floor)
        
        # Generate full autocorrelation for plotting
        h = (sig - np.mean(sig)) / (np.linalg.norm(sig) + 1e-12)
        corr = np.correlate(h, h, mode='full') / (noise_floor + 1e-12)
        lags = np.arange(-len(sig)+1, len(sig)) * dt * 1e6  # μs
        
        plt.plot(lags, corr, 
                label=f"{lbl} (Peak={peak:.1f}σ, Width={width*dt*1e6:.1f}μs, PSR={psr:.1f}dB)")
    
    plt.xlabel("Lag (μs)")
    plt.ylabel("Correlation (σ units)")
    plt.title("Autocorrelation Comparison")
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(save_dir, "autocorrelation_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved autocorrelation comparison to: {save_path}")

def compare_sensor_signals(info_dict, sensitivity_map, weighted_p, match_filter_results, sim_params, save_dir):
    """
    Main comparison function with proper physical units.
    """
    # Get noise floor from matched filter results
    noise_floor = np.std(match_filter_results['noise_only'])
    
    # Find extreme sensitivity points
    max_pos, min_pos, sensor_idx_max, sensor_idx_min = find_sensitivity_extremes(
        info_dict, sensitivity_map)
    
    # Extract signals and ensure proper shape
    center_signal = weighted_p[:, sensor_idx_max].flatten()
    edge_signal = weighted_p[:, sensor_idx_min].flatten()
    
    # Ensure time array is properly shaped
    t_array = sim_params['kgrid'].t_array.flatten()
    
    # Compare signals
    compare_signals_plot(
        signals=[center_signal, edge_signal],
        t_array=t_array,
        labels=["Center (Max Sens)", "Edge (Min Sens)"],
        save_dir=save_dir,
        noise_floor=noise_floor,
        normalize=True
    )
    
    # Compare autocorrelations
    compare_autocorrelations(
        signals=[center_signal, edge_signal],
        labels=["Center", "Edge"],
        save_dir=save_dir,
        noise_floor=noise_floor,
        t_array=t_array
    )
    
    # Calculate metrics
    center_peak, center_width, center_psr = calculate_autocorr_metrics(center_signal, noise_floor)
    edge_peak, edge_width, edge_psr = calculate_autocorr_metrics(edge_signal, noise_floor)
    
    return {
        'center_snr_db': 10*np.log10(np.mean(center_signal**2)/(noise_floor**2 + 1e-12)),
        'edge_snr_db': 10*np.log10(np.mean(edge_signal**2)/(noise_floor**2 + 1e-12)),
        'center_autocorr_width_us': center_width * (sim_params['kgrid'].dt * 1e6),
        'edge_autocorr_width_us': edge_width * (sim_params['kgrid'].dt * 1e6),
        'center_psr_db': center_psr,
        'edge_psr_db': edge_psr
    }