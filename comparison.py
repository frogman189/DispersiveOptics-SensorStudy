import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import correlate
from post_processing import find_sensitivity_extremes

# def compare_signals_plot(signals: list, t_array: np.ndarray, labels: list, save_dir: str,
#                     normalize=True, sigma_noise: float = None):
#     plt.figure(figsize=(10, 6))
#     t = np.ravel(t_array)

#     y_max = 0.0
#     curves = []
#     for sig, lbl in zip(signals, labels):
#         if sigma_noise is not None:
#             snr_peak = np.max(np.abs(sig)) / sigma_noise
#             sig_plot = sig / sigma_noise
#             label = f"{lbl} (SNR_peak={snr_peak:.1f}σ)"
#             normalize_here = False
#         else:
#             sig_plot = sig.copy()
#             label = lbl
#             normalize_here = normalize

#         if normalize_here:
#             sig_plot = (sig_plot - np.min(sig_plot)) / (np.max(sig_plot) - np.min(sig_plot) + 1e-12)

#         y_max = max(y_max, np.max(np.abs(sig_plot)))
#         curves.append((sig_plot, label))

#     for sig_plot, label in curves:
#         plt.plot(t*1e6, sig_plot, label=label)

#     plt.xlabel("Time (μs)")
#     plt.ylabel("Pressure (σ-units)" if sigma_noise is not None else "Normalized Pressure")
#     plt.title("Signal Comparison")
#     plt.legend()
#     plt.grid(True)
#     if sigma_noise is not None:
#         plt.ylim(-1.1*y_max, 1.1*y_max)

#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, "signal_comparison.png")
#     plt.savefig(save_path)
#     plt.close()
#     print(f"Saved signal comparison to: {save_path}")



# def compare_autocorrelations(signals: list, labels: list, save_dir: str,
#                              sigma_noise: float, t_array: np.ndarray):
#     """
#     Compare autocorrelation-like matched outputs in σ-units.
#     For each signal s, use a unit-energy template h = s / ||s||2 (autocorrelation proxy).
#     Plot y = (s_noisy ⋆ h) / sigma_noise so peak height is directly comparable.
#     """
#     plt.figure(figsize=(10, 6))
#     t = np.ravel(t_array)
#     dt = t[1] - t[0]

#     for sig, lbl in zip(signals, labels):
#         # unit-energy template
#         h = sig - np.mean(sig)
#         h /= (np.linalg.norm(h) + 1e-12)

#         # Noisy version for a fair comparison (same sigma)
#         noisy = sig + np.random.normal(0.0, sigma_noise, size=sig.shape)

#         y = correlate(noisy - np.mean(noisy), h, mode='full') / (sigma_noise + 1e-12)
#         lags = np.arange(-len(sig)+1, len(sig)) * dt  # seconds

#         peak = np.max(np.abs(y))
#         # crude -3 dB width and PSL for info
#         half = peak / np.sqrt(2)
#         pk_idx = np.argmax(np.abs(y))
#         L = pk_idx
#         while L > 0 and np.abs(y[L]) >= half: L -= 1
#         R = pk_idx
#         while R < len(y)-1 and np.abs(y[R]) >= half: R += 1
#         mainlobe_us = (R - L) * dt * 1e6

#         guard = 5
#         mask = np.ones_like(y, dtype=bool)
#         mask[max(0, pk_idx-guard):min(len(y), pk_idx+guard+1)] = False
#         psl_db = 20*np.log10((np.max(np.abs(y[mask])) + 1e-12) / (peak + 1e-12))

#         plt.plot(lags*1e6, y, label=f"{lbl} (peak={peak:.1f}σ, width≈{mainlobe_us:.1f}µs, PSL={psl_db:.1f} dB)")

#     plt.xlabel("Lag (μs)")
#     plt.ylabel("Correlation (σ-units)")
#     plt.title("Autocorrelation / Matched Output Comparison")
#     plt.legend()
#     plt.grid(True)
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, "autocorrelation_comparison.png")
#     plt.savefig(save_path)
#     plt.close()
#     print(f"Saved autocorrelation comparison to: {save_path}")


# def nearest_and_farthest_true(mask: np.ndarray):
#     """
#     Given a 2D boolean mask, find the nearest and farthest True indices from the mask center.
    
#     Parameters
#     ----------
#     mask : np.ndarray
#         2D boolean array where True indicates the sensor shape.
    
#     Returns
#     -------
#     nearest_idx : tuple
#         (row, col) index of the True pixel nearest to the center.
#     farthest_idx : tuple
#         (row, col) index of the True pixel farthest from the center.
#     """
#     if mask.ndim != 2 or mask.dtype != bool:
#         raise ValueError("mask must be a 2D boolean numpy array")

#     # Calculate geometric center of the array
#     center_row = (mask.shape[0] - 1) / 2
#     center_col = (mask.shape[1] - 1) / 2
#     center = np.array([center_row, center_col])

#     # Get coordinates of all True pixels
#     true_coords = np.argwhere(mask)  # shape (N, 2)

#     if true_coords.size == 0:
#         raise ValueError("mask contains no True values")

#     # Compute distances from center
#     distances = np.linalg.norm(true_coords - center, axis=1)

#     # Find nearest and farthest
#     nearest_idx = tuple(true_coords[np.argmin(distances)])
#     farthest_idx = tuple(true_coords[np.argmax(distances)])

#     return nearest_idx, farthest_idx


# def compare_sensor_signals(info_dict, sensitivity_map, weighted_p, match_filter_results, sim_params, save_dir):

#     sigma_noise = np.std(match_filter_results['noise_only'])

#     max_pos, min_pos, sensor_idx_max, sensor_idx_min = find_sensitivity_extremes(info_dict, sensitivity_map)
#     center_signal = weighted_p[:, sensor_idx_max]
#     edge_signal = weighted_p[:, sensor_idx_min]

#     # Make noisy versions with the SAME sigma (fair comparison)
#     center_noisy = center_signal + np.random.normal(0.0, sigma_noise, size=center_signal.shape)
#     edge_noisy   = edge_signal   + np.random.normal(0.0, sigma_noise, size=edge_signal.shape)

#     # Compare in σ-units (no min-max)
#     compare_signals_plot(
#         signals=[center_noisy, edge_noisy],
#         t_array=sim_params['kgrid'].t_array,
#         labels=["Center", "Edge"],
#         save_dir=save_dir,
#         normalize=False,
#         sigma_noise=sigma_noise
#     )

#     compare_autocorrelations(
#         signals=[center_signal, edge_signal],   # templates from the CLEAN signals
#         labels=["Center", "Edge"],
#         save_dir=save_dir,
#         sigma_noise=sigma_noise,
#         t_array=sim_params['kgrid'].t_array
#     )
    
#     # Peak SNR in dB (amplitude-based)
#     center_snr_db = 20*np.log10(np.max(np.abs(center_noisy)) / sigma_noise)
#     edge_snr_db   = 20*np.log10(np.max(np.abs(edge_noisy))   / sigma_noise)

#     return {
#         'center_snr_db': center_snr_db,
#         'edge_snr_db': edge_snr_db
#     }


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