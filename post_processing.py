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


def find_sensitivity_extremes(sensitivity_map: np.ndarray):
    """
    Find the position of the maximum value and the minimum non-zero value
    in a 2D sensitivity_map. If multiple positions tie, returns the first one.

    Args:
        sensitivity_map: 2D numpy array of sensitivities.

    Returns:
        max_pos: Tuple[int, int] for the location of the global maximum.
        min_non_zero_pos: Tuple[int, int] for the location of the minimum non-zero value,
                          or None if there are no non-zero entries.
    """
    # --- Maximum ---
    max_val = sensitivity_map.max()
    max_positions = np.argwhere(sensitivity_map == max_val)
    max_pos = tuple(max_positions[0])  # first occurrence

    # --- Minimum non-zero ---
    non_zero_mask = sensitivity_map != 0
    if non_zero_mask.any():
        non_zero_vals = sensitivity_map[non_zero_mask]
        min_non_zero_val = non_zero_vals.min()
        min_positions = np.argwhere(sensitivity_map == min_non_zero_val)
        min_non_zero_pos = tuple(min_positions[0])  # first occurrence
    else:
        min_non_zero_pos = None

    return max_pos, min_non_zero_pos




def analyze_sensor_performance(p_data, t_array, sensor_mask=None, sensitivity_map=None, weighted_p=None):
    """
    Compute performance metrics for a sensor's pressure recordings.
    
    Args:
        p_data (np.ndarray): Pressure data array (shape: Nt or Nt x N_sensor_points)
        t_array (np.ndarray): Time array (shape: Nt,)
        sensor_mask (np.ndarray, optional): Boolean mask of sensor points
        sensitivity_map (np.ndarray, optional): Sensitivity weights matching sensor_mask
        
    Returns:
        dict: Dictionary of metrics including time-wise sums across all sensor points
    """
    # Ensure pressure data is 2D (even for single sensor)
    if p_data.ndim == 1:
        p_data = p_data.reshape(-1, 1)
    
    Nt, N_points = p_data.shape
    dt = t_array[1] - t_array[0]
    
    # Initialize weighted pressure array (if sensitivity provided)
    if weighted_p is not None:
        p_weighted = weighted_p
    elif sensitivity_map is not None and sensor_mask is not None:
        mask_flat = sensor_mask.flatten(order='F')
        sens_flat = sensitivity_map.flatten(order='F')
        sensor_idx = np.where(mask_flat == 1)[0]
        sens_vals = sens_flat[sensor_idx] 
        p_weighted = p_data * sens_vals[np.newaxis, :]
    else:
        p_weighted = None
    
    # Time-wise sums across all sensor points
    p_total = np.sum(p_data, axis=1)            # Sum of raw pressures (shape: Nt)
    p_total_weighted = np.sum(p_weighted, axis=1) if p_weighted is not None else None
    
    # Frequency domain setup
    freqs = fftfreq(Nt, dt)
    
    metrics = {
        'point_metrics': [],           # Metrics for each individual sensor point
        'aggregate_metrics': {},       # Statistics across all sensor points
        'time_series_metrics': {       # Time-wise summed signals
            'p_total': p_total,
            'p_total_weighted': p_total_weighted
        }
    }
    
    # Process each sensor point (original logic)
    for i in range(N_points):
        p = p_data[:, i]
        p_w = p_weighted[:, i] if p_weighted is not None else None
        
        # Time-domain metrics (unchanged)
        peaks, _ = find_peaks(np.abs(p), height=0.5*np.max(np.abs(p)))
        if len(peaks) == 0:
            first_peak_idx = last_peak_idx = np.argmax(np.abs(p))
        else:
            first_peak_idx = peaks[0]
            last_peak_idx = peaks[-1]
        
        point_metrics = {
            'peak_pressure': np.max(np.abs(p)),
            'time_to_peak': t_array[np.argmax(np.abs(p))],
            'peak_to_peak': np.ptp(p),
            'energy': trapezoid(p**2, t_array),
            'weighted_energy': trapezoid(p_w**2, t_array) if p_w is not None else None,
            'signal_duration': t_array[last_peak_idx] - t_array[first_peak_idx],
            'dominant_frequency': freqs[np.argmax(np.abs(fft(p)))],
            'rms_pressure': np.sqrt(np.mean(p**2))
        }
        metrics['point_metrics'].append(point_metrics)
    
    # Aggregate statistics (unchanged)
    peak_pressures = [m['peak_pressure'] for m in metrics['point_metrics']]
    energies = [m['energy'] for m in metrics['point_metrics']]
    
    metrics['aggregate_metrics'] = {
        'mean_peak_pressure': np.mean(peak_pressures),
        'std_peak_pressure': np.std(peak_pressures),
        'max_peak_pressure': np.max(peak_pressures),
        'total_energy': np.sum(energies),
        'energy_uniformity': np.std(energies) / np.mean(energies),
        'mean_time_to_peak': np.mean([m['time_to_peak'] for m in metrics['point_metrics']])
    }
    
    return metrics