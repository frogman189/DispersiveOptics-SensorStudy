import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.integrate import trapezoid
from scipy.signal import find_peaks
from scipy.signal import correlate


def process_sensor_data(sensor_mask: np.ndarray,
                        sensitivity_map: np.ndarray,
                        p: np.ndarray):
    """
    Parameters
    ----------
    sensor_mask : (Nx, Ny) binary array
        1’s mark the sensor locations.
    sensitivity_map : (Nx, Ny) array
        Sensitivity values aligned with sensor_mask.
    p : (n_time, n_sensor_points) array
        Recorded pressure time series for each sensor point.

    Returns
    -------
    mask_flat : (Nx*Ny,) array
        sensor_mask flattened in Fortran (column-major) order.
    weighted_p : (n_time, n_sensor_points) array
        p multiplied by each sensor’s sensitivity.
    info_dict : dict[int → dict]
        For each sensor index j (0 ≤ j < n_sensor_points):
          • 'p_before'    : (n_time,) raw series p[:, j]
          • 'sensitivity' : scalar sensitivity_map value
          • 'weighted_p'  : (n_time,) p_before * sensitivity
          • 'sensor_index': j
          • 'grid_index'  : (row, col) in the original mask
    """
    # 1) flatten mask & sensitivity in column-major order
    mask_flat = sensor_mask.flatten(order='F')
    sens_flat = sensitivity_map.flatten(order='F')

    # 2) find which flattened indices are sensors
    sensor_idx = np.where(mask_flat == 1)[0]
    n_sensors = len(sensor_idx)

    # 3) pull out the sensitivity values in the correct order
    sens_vals = sens_flat[sensor_idx]                     # (n_sensor_points,)

    # 4) weight the entire p array
    weighted_p = p * sens_vals[np.newaxis, :]             # broadcast multiply

    # 5) build the info dict
    info_dict = {}
    for j, flat_idx in enumerate(sensor_idx):
        row, col = np.unravel_index(flat_idx, sensor_mask.shape, order='F')
        p_before = p[:, j]                                # raw time series
        info_dict[j] = {
            'p_before':    p_before,
            'sensitivity': sens_vals[j],
            'weighted_p':  weighted_p[:, j],
            'sensor_index': j,
            'grid_index':  (row, col)
        }

    return mask_flat, weighted_p, info_dict


def find_sensitivity_extremes(info_dict, sensitivity_map: np.ndarray):
    """
    Given a 2D sensitivity_map array, find:
      1. max_pos: the (row, col) of the global maximum value; 
         if there are ties, pick the one closest to the center of the map.
      2. min_pos: the (row, col) of the smallest nonzero value (excluding the max value);
         if there are ties, pick the one farthest from the center.
    
    Returns:
        max_pos: tuple of ints (row, col) or None if map is empty
        min_pos: tuple of ints (row, col) or None if no valid nonzero minima exist
    """
    h, w = sensitivity_map.shape
    center = np.array([h / 2, w / 2])

    # 1) Find max positions
    max_val = np.nanmax(sensitivity_map)
    max_positions = np.argwhere(sensitivity_map == max_val)
    # compute distances to center, pick the closest
    dists_to_center = np.linalg.norm(max_positions - center, axis=1)
    max_idx = np.argmin(dists_to_center)
    max_pos = tuple(max_positions[max_idx])

    # 2) Find min non-zero (and not equal to max_val) positions
    mask = (sensitivity_map != 0)
    if np.any(mask):
        nonzero_vals = sensitivity_map[mask]
        min_val = nonzero_vals.min()
        min_positions = np.argwhere(sensitivity_map == min_val)
        # compute distances, pick the farthest
        dists_min = np.linalg.norm(min_positions - center, axis=1)
        min_idx = np.argmax(dists_min)
        min_pos = tuple(min_positions[min_idx])

        sensor_idx_min = next(
            (idx for idx, data in info_dict.items() if data['grid_index'] == min_pos),
            None
        )
    else:
        min_pos = None
        sensor_idx_min = None
        
    sensor_idx_max = next(
        (idx for idx, data in info_dict.items() if data['grid_index'] == max_pos),
        None
    )

    return max_pos, min_pos, sensor_idx_max, sensor_idx_min


def plot_raw_pressure(raw_data, weighted_data, kgrid, sensor_indices):
    """
    Compare raw and weighted pressure at multiple sensor points,
    plotting each sensor in its own subplot.
    
    Args:
        raw_data: dict with key 'p' of shape (Nt,) or (Nt, n_sensors)
        weighted_data: same shape as raw_data['p']
        kgrid: object with t_array attribute (1×Nt or Nt×1)
        sensor_indices: list of int sensor indices to plot
    """
    plots_names = ["Pressure over time in max sensor point", "Pressure over time in max sensor point"]
    t = kgrid.t_array.flatten()
    n = len(sensor_indices)
    n = sum(x is not None for x in sensor_indices)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, sensor_idx, plot_name in zip(axes, sensor_indices, plots_names):
        if (sensor_idx is not None):
            # pull out 1D traces
            if raw_data['p'].ndim == 1:
                raw_p      = raw_data['p']
                weighted_p = weighted_data
            else:
                raw_p      = raw_data['p'][:, sensor_idx]
                weighted_p = weighted_data[:, sensor_idx]

            ax.plot(t, raw_p,      label='Raw',     alpha=0.5)
            #ax.plot(t, weighted_p, '--', label='Weighted')

            ax.set_title(f'{plot_name}')
            ax.set_xlabel('Time (s)')
            ax.grid(True)
            if ax is axes[0]:
                ax.set_ylabel('Pressure (Pa)')
            ax.legend()

    plt.tight_layout()
    plt.show()




def analyze_sensor_performance(p_data, t_array, sensor_mask=None, sensitivity_map=None, weighted_p=None):
    """
    Compute performance metrics for a sensor's pressure recordings using ONLY weighted pressure.
    
    Args:
        p_data (np.ndarray): Pressure data array (shape: Nt or Nt x N_sensor_points)
        t_array (np.ndarray): Time array (shape: Nt,)
        sensor_mask (np.ndarray, optional): Boolean mask of sensor points
        sensitivity_map (np.ndarray, optional): Sensitivity weights matching sensor_mask
        
    Returns:
        dict: Dictionary of metrics calculated using weighted pressure only
    """
    # Ensure pressure data is 2D (even for single sensor)
    if p_data.ndim == 1:
        p_data = p_data.reshape(-1, 1)
    
    Nt, N_points = p_data.shape
    dt = t_array[1] - t_array[0]
    
    # Initialize weighted pressure array (required)
    if weighted_p is not None:
        p_weighted = weighted_p
    elif sensitivity_map is not None and sensor_mask is not None:
        mask_flat = sensor_mask.flatten(order='F')
        sens_flat = sensitivity_map.flatten(order='F')
        sensor_idx = np.where(mask_flat == 1)[0]
        sens_vals = sens_flat[sensor_idx] 
        p_weighted = p_data * sens_vals[np.newaxis, :]
    else:
        raise ValueError("Either weighted_p or sensitivity_map+sensor_mask must be provided")
    
    # Time-wise sum of weighted pressures
    p_total_weighted = np.sum(p_weighted, axis=1)
    
    # Frequency domain setup
    freqs = fftfreq(Nt, dt)
    
    metrics = {
        'point_metrics': [],           # Metrics for each sensor point
        'aggregate_metrics': {},       # Statistics across all sensor points
        'time_series_metrics': {       # Time-wise summed signals
            'p_total_weighted': p_total_weighted
        }
    }
    
    # Process each sensor point using weighted pressure only
    for i in range(N_points):
        p_w = p_weighted[:, i]
        
        # Find peaks in weighted pressure
        peaks, _ = find_peaks(np.abs(p_w), height=0.5*np.max(np.abs(p_w)))
        if len(peaks) == 0:
            first_peak_idx = last_peak_idx = np.argmax(np.abs(p_w))
        else:
            first_peak_idx = peaks[0]
            last_peak_idx = peaks[-1]
        
        point_metrics = {
            'weighted_peak_pressure': np.max(np.abs(p_w)),
            'weighted_time_to_peak': t_array[np.argmax(np.abs(p_w))],
            'weighted_peak_to_peak': np.ptp(p_w),
            'weighted_energy': trapezoid(p_w**2, t_array),
            'weighted_signal_duration': t_array[last_peak_idx] - t_array[first_peak_idx],
            'weighted_dominant_frequency': freqs[np.argmax(np.abs(fft(p_w)))],
            'weighted_rms_pressure': np.sqrt(np.mean(p_w**2))
        }
        metrics['point_metrics'].append(point_metrics)
    
    # Aggregate statistics using weighted values only
    weighted_peak_pressures = [m['weighted_peak_pressure'] for m in metrics['point_metrics']]
    weighted_energies = [m['weighted_energy'] for m in metrics['point_metrics']]
    
    metrics['aggregate_metrics'] = {
        'mean_peak_pressure': np.mean(weighted_peak_pressures),
        'std_peak_pressure': np.std(weighted_peak_pressures),
        'max_peak_pressure': np.max(weighted_peak_pressures),
        'total_energy': np.sum(weighted_energies),
        'energy_uniformity': np.std(weighted_energies) / np.mean(weighted_energies),
        'mean_time_to_peak': np.mean([m['weighted_time_to_peak'] for m in metrics['point_metrics']])
    }
    
    return metrics



def print_detector_metrics(metrics: dict, detection_comparison: dict = None):
    """
    Prints detector metrics and detection thresholds in a standardized format.
    
    Args:
        metrics: Dictionary from calculate_detector_metrics()
                 (contains NEP, D*, noise_floor, responsivity, signal_response)
        detection_comparison: Dictionary from compare_detection_thresholds()
                             (contains raw/filtered thresholds and SNRs)
    """
    print("\n=== DETECTOR PERFORMANCE METRICS ===")
    print(f"{'Noise Equivalent Pressure (NEP)':<35}: {metrics['NEP']:.2e} Pa")
    print(f"{'Detectivity (D*)':<35}: {metrics['D_star']:.2e} Jones")
    print(f"{'Noise Floor':<35}: {metrics['noise_floor']:.2e} Pa")
    print(f"{'Responsivity':<35}: {metrics['responsivity']:.2e} (output/Pa)")
    print(f"{'Signal Response':<35}: {metrics['signal_response']:.2e} Pa")
    
    if detection_comparison:
        print("\n=== DETECTION THRESHOLD COMPARISON ===")
        print(f"{'Raw Detection Threshold (5σ)':<35}: {detection_comparison['raw_threshold_pa']:.2e} Pa")
        print(f"{'Filtered Detection Threshold (5σ)':<35}: {detection_comparison['filtered_threshold_pa']:.2e} Pa")
        print(f"{'SNR Improvement':<35}: {detection_comparison['raw_snr']:.1f} → {detection_comparison['filtered_snr']:.1f}")
        print(f"{'Effective Improvement Factor':<35}: {detection_comparison['improvement_factor']:.1f}x")
    print("="*50)
