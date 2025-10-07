import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import correlate
from scipy.signal import correlate, correlation_lags


def add_gaussian_noise(signal: np.ndarray,
                       snr_db: float = 10) -> dict:
    """
    Add zero-mean Gaussian noise to a 1D signal at a specified SNR (dB) and return detailed components.

    Steps:
      1. Compute the power of the original signal (mean square).
      2. Determine the noise power needed for the specified SNR.
      3. Generate Gaussian noise with zero mean and computed variance.
      4. Add the noise to the original signal.

    Args:
        signal (np.ndarray): Input signal array.
        snr_db (float): Desired signal-to-noise ratio in decibels.

    Returns:
        dict: {
            'noisy_signal': np.ndarray, the signal with added noise;
            'noise': np.ndarray, the generated noise array;
            'signal_power': float, computed power of the original signal;
            'noise_power': float, computed power of the noise;
        }
    """
    # Compute signal power
    signal_power = np.mean(signal ** 2)
    # Determine noise power from SNR (linear scale)
    noise_power = signal_power / (10 ** (snr_db / 10))

    # Generate zero-mean Gaussian noise
    noise = np.random.normal(loc=0.0,
                             scale=np.sqrt(noise_power),
                             size=signal.shape)

    # Create noisy signal
    noisy_signal = signal + noise

    return noisy_signal

def add_noise_fixed_sigma(signal: np.ndarray, sigma=100) -> np.ndarray:
    noise = np.random.normal(loc=0.0, scale=sigma, size=signal.shape)
    return signal + noise


def plot_presure_and_noise_presure_over_time(p, kgrid, snr_db=10, save_to_dir=False, output_dir=None):
    """
    Plot the summed raw pressure and noise-added pressure over time, side by side.

    Args:
        p (ndarray): Array of shape (Nt, ...) containing raw pressures.
        kgrid: Object with attribute t_array of shape (1, Nt) or (Nt,).
        snr_db (float): Desired signal-to-noise ratio in dB for noise addition.
        save_to_dir (bool): If True, saves plot to output_dir instead of showing.
        output_dir (str): Directory path to save the plot when save_to_dir=True.
    """
    
    # Sum over spatial axes to get time series (shape: Nt)
    p_total = np.sum(p, axis=tuple(range(1, p.ndim)))
    #p_total_noisy = add_gaussian_noise(p_total, snr_db=snr_db)
    p_total_noisy = add_noise_fixed_sigma(p_total, sigma=100)
    # Flatten time axis
    t = kgrid.t_array.flatten()

    # Create side-by-side plots
    plt.figure(figsize=(12, 4))

    # Raw pressure
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(t, p_total)
    ax1.set_title("Raw Pressure vs Time")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Summed Pressure")

    # Noisy pressure
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(t, p_total_noisy)
    ax2.set_title(f"Noisy Pressure (SNR = {snr_db} dB)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Summed Pressure")

    plt.tight_layout()
    
    if save_to_dir and output_dir:
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        # Create filename
        filename = f"pressure_plot.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()




def apply_match_filter(p: np.ndarray, kgrid, sigma_noise: float, win_samples: int = 200, remove_dc: bool = True) -> dict:
    """
    Apply a matched filter to a spatially-summed pulse and compute **comparable** detection SNRs.

    Comparable metrics:
        Raw SNR (RMS):  RMS(signal over template window) / σ
        MF SNR:         peak(|corr(received, unit_energy_template)|) / σ
    In AWGN and perfect template match, MF gain ≈ √M where M = win_samples.

    Args:
        p: Pulse data (Nt, *spatial*). Will be summed over spatial dims.
        kgrid: Has t_array (Nt,).
        sigma_noise: Std of added white Gaussian noise (same for all sensors).
        win_samples: Template length M (samples) centered on the clean peak.
        remove_dc: Subtract mean from template and received before processing.

    Returns: dict with signals, indices, SNRs, etc.
    """
    # 1) Collapse spatial dims
    p_total = np.sum(p, axis=tuple(range(1, p.ndim)))  # (Nt,)
    t = np.asarray(kgrid.t_array).reshape(-1)
    dt = float(t[1] - t[0])

    # 2) Add white noise
    rng = np.random.default_rng()
    noise_only = rng.normal(0.0, sigma_noise, size=p_total.shape)
    p_noisy = p_total + noise_only

    # 3) Template window around CLEAN peak (to avoid noise jitter in picking the window)
    peak_idx_clean = int(np.argmax(np.abs(p_total)))
    M = int(win_samples)
    half = M // 2
    i0 = max(0, peak_idx_clean - half)
    i1 = min(len(p_total), i0 + M)
    if (i1 - i0) < M:
        i0 = max(0, i1 - M)

    template = p_total[i0:i1].astype(float).copy()
    if remove_dc:
        template -= np.mean(template)
    E = float(np.dot(template, template))
    if E <= 0:
        raise RuntimeError("Template has zero energy; try a different window.")
    template /= np.sqrt(E)  # unit-energy

    # 4) Prepare received (optional DC removal using a baseline estimate)
    received = p_noisy.astype(float)
    if remove_dc:
        # subtract a global mean (or use a known pre-event baseline if you have indices)
        received -= np.mean(received[: max(8*half, M)])  # robust-ish baseline removal

    # 5) Matched filtering (cross-correlation with unit-energy template)
    corr_full = correlate(received, template, mode='full')
    lags = correlation_lags(len(received), len(template), mode='full')
    mask_full = (lags >= 0) & (lags <= (len(received) - M))
    corr_use = corr_full[mask_full]
    lags_use = lags[mask_full]
    lags_s = lags_use * dt

    peak_idx_mf = int(np.argmax(np.abs(corr_use)))
    y_peak = float(np.abs(corr_use[peak_idx_mf]))
    t_detect = float(lags_s[peak_idx_mf])

    # === CONSISTENT NOISE STDs ===
    # For unit-energy template and white noise, σ_mf == σ_raw == sigma_noise
    # Use the known sigma to avoid contamination from signal sidelobes.
    sigma_raw = float(np.std(noise_only, ddof=1))  # sanity check; ~ sigma_noise
    sigma_mf = sigma_noise

    # === COMPARABLE SNRs ===
    # Raw SNR: RMS of the (CLEAN) signal over the SAME window M divided by σ
    # (Using clean signal to define the signal energy; detection is on noisy data anyway.)
    s_win = p_total[i0:i1].astype(float)
    if remove_dc:
        s_win = s_win - np.mean(s_win)
    rms_signal = float(np.sqrt(np.mean(s_win**2)))
    raw_peak = np.max(np.abs(p_noisy[i0:i1]))
    raw_detection_snr = raw_peak / sigma_raw
    #raw_detection_snr = rms_signal / (sigma_raw + 1e-12)

    # MF SNR: correlation peak divided by σ (since unit-energy template)
    mf_detection_snr = y_peak / (sigma_mf + 1e-12)

    snr_improvement = mf_detection_snr / (raw_detection_snr + 1e-12)
    print(f"SNR raw: ", raw_detection_snr)
    print(f"SNR MF: ", mf_detection_snr)
    print(f"SNR improvement: ", snr_improvement)

    # Optional: empirical corr-domain noise check (exclude a guard around the peak)
    guard = max(M, 100)
    mask_noise = np.ones_like(corr_use, dtype=bool)
    mask_noise[np.maximum(0, peak_idx_mf-guard): np.minimum(len(corr_use), peak_idx_mf+guard+1)] = False
    if np.any(mask_noise):
        sigma_mf_emp = float(np.std(corr_use[mask_noise], ddof=1))
    else:
        sigma_mf_emp = np.nan  # not enough samples

    return {
        'noise_only': noise_only,
        'clean_pulse': p_total,
        'received': received,
        'lags': lags_s,
        'matched_output': corr_use,
        'detection_time': t_detect,
        'template': template,
        'indices': {'i0': i0, 'i1': i1, 'peak_idx_clean': peak_idx_clean, 'peak_idx_mf': peak_idx_mf},
        'sigma_raw': sigma_raw,
        'sigma_mf_theory': sigma_mf,
        'sigma_mf_empirical': sigma_mf_emp,
        'raw_detection_snr': raw_detection_snr,
        'mf_detection_snr': mf_detection_snr,
        'snr_improvement': snr_improvement,
        'raw_threshold_5sigma': 5.0 * sigma_raw,
        'mf_threshold_5sigma': 5.0 * sigma_mf,
    }




def plot_match_filter_results(results: dict,
                            t: np.ndarray,
                            snr_db: float,
                            save_to_dir=False, output_dir=None):
    
    # Unpack results
    noise_only     = results['noise_only']
    clean_pulse    = results['clean_pulse']
    received       = results['received']
    lags_pos       = results['lags']
    corr_pos       = results['matched_output']
    t_detect       = results['detection_time']

    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    # Top-left: Noise only
    axs[0, 0].plot(t * 1e6, noise_only, color='gray')
    axs[0, 0].set_title(f'Noise only (SNR={snr_db} dB)')
    axs[0, 0].set_ylabel('Amplitude')
    axs[0, 0].grid(True)
    # Top-right: Clean pulse
    axs[0, 1].plot(t * 1e6, clean_pulse, 'g')
    axs[0, 1].set_title('Clean signal')
    axs[0, 1].grid(True)
    # Bottom-left: Noisy & matched
    axs[1, 0].plot(t * 1e6, received, 'b', label='Noisy')
    axs[1, 0].plot(lags_pos * 1e6, corr_pos, 'r', label='Matched')
    axs[1, 0].axvline(t_detect * 1e6, color='k', ls='--',
                      label=f'Peak @ {t_detect*1e6:.2f} µs')
    axs[1, 0].set_title('Noisy & Matched Output')
    axs[1, 0].set_xlabel('Time (µs)')
    axs[1, 0].set_ylabel('Amplitude')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    # Bottom-right: Clean & matched
    axs[1, 1].plot(t * 1e6, clean_pulse, 'g', label='Clean')
    axs[1, 1].plot(lags_pos * 1e6, corr_pos, 'r', label='Matched')
    axs[1, 1].set_title('Clean & Matched Output')
    axs[1, 1].set_xlabel('Time (µs)')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    if save_to_dir and output_dir:
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        # Create filename
        filename = f"match_filter_results.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    



def save_match_filter_plots(results: dict,
                            t: np.ndarray,
                            snr_db: float,
                            output_dir: str,
                            sensor_type: int,
                            sensor_index: int) -> str:
    """
    Plot a 2×2 grid of matched-filter outputs and save to disk under a sensor-specific folder.

    Args:
        results (dict): Output of apply_match_filter with keys:
            'noise_only', 'clean_pulse', 'received', 'lags', 'matched_output', 'detection_time'.
        t (np.ndarray): Time vector (s) for the signals (length Nt).
        snr_db (float): SNR in dB used to generate noise.
        output_dir (str): Base directory to save plots.
        sensor_type (int): Code for sensor geometry (1–8).
        sensor_index (int): Index of the sensor point for labeling the file.

    Returns:
        str: Full path of the saved plot image.
    """
    # Factory mapping for sensor_type
    SENSOR_FACTORY = {
        1: 'point',
        2: 'linear_delta',
        3: 'linear_gaussian',
        4: 'linear_uniform',
        5: 'rectangular_gaussian',
        6: 'rectangular_uniform',
        7: 'snake_linear',
        8: 'rect_checkerboard'
    }
    sensor_name = SENSOR_FACTORY.get(sensor_type, f'type_{sensor_type}')
    save_dir = os.path.join(output_dir, sensor_name)
    os.makedirs(save_dir, exist_ok=True)

    # Unpack results
    noise_only     = results['noise_only']
    clean_pulse    = results['clean_pulse']
    received       = results['received']
    lags_pos       = results['lags']
    corr_pos       = results['matched_output']
    t_detect       = results['detection_time']

    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    # Top-left: Noise only
    axs[0, 0].plot(t * 1e6, noise_only, color='gray')
    axs[0, 0].set_title(f'Noise only (SNR={snr_db} dB)')
    axs[0, 0].set_ylabel('Amplitude')
    axs[0, 0].grid(True)
    # Top-right: Clean pulse
    axs[0, 1].plot(t * 1e6, clean_pulse, 'g')
    axs[0, 1].set_title('Clean signal')
    axs[0, 1].grid(True)
    # Bottom-left: Noisy & matched
    axs[1, 0].plot(t * 1e6, received, 'b', label='Noisy')
    axs[1, 0].plot(lags_pos * 1e6, corr_pos, 'r', label='Matched')
    axs[1, 0].axvline(t_detect * 1e6, color='k', ls='--',
                      label=f'Peak @ {t_detect*1e6:.2f} µs')
    axs[1, 0].set_title('Noisy & Matched Output')
    axs[1, 0].set_xlabel('Time (µs)')
    axs[1, 0].set_ylabel('Amplitude')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    # Bottom-right: Clean & matched
    axs[1, 1].plot(t * 1e6, clean_pulse, 'g', label='Clean')
    axs[1, 1].plot(lags_pos * 1e6, corr_pos, 'r', label='Matched')
    axs[1, 1].set_title('Clean & Matched Output')
    axs[1, 1].set_xlabel('Time (µs)')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()

    # Save figure
    filename = f"sensor_{sensor_index}.png"
    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"Saved matched-filter plot to: {save_path}")
    return save_path


def calculate_detector_metrics(p_data: np.ndarray,
                             noise_data: np.ndarray,
                             source_pressure: float,
                             detector_area: float,
                             bandwidth: float) -> dict:
    """
    Calculate fundamental detector performance metrics for acoustic sensors.
    
    Args:
        p_data: Pressure data array (time × spatial dims) in Pascals
        noise_data: Background noise data (time × spatial dims) in Pascals
        source_pressure: Known source pressure amplitude in Pascals
        detector_area: Effective detector area in m²
        bandwidth: System bandwidth in Hz
        
    Returns:
        Dictionary containing:
            - 'NEP': Noise Equivalent Pressure (Pa)
            - 'D_star': Detectivity (Jones)
            - 'noise_floor': RMS noise level (Pa)
            - 'responsivity': Sensor responsivity
    """
    # Calculate RMS noise floor
    noise_floor = np.std(noise_data)
    
    # Calculate peak signal response (above noise mean)
    signal_response = np.max(p_data) - np.mean(noise_data)
    
    # Calculate responsivity (output per input pressure)
    responsivity = signal_response / source_pressure
    
    # Calculate NEP (Noise Equivalent Pressure)
    NEP = noise_floor / responsivity
    
    # Calculate D* (Detectivity)
    D_star = np.sqrt(detector_area * bandwidth) / NEP
    
    return {
        'NEP': NEP,
        'D_star': D_star,
        'noise_floor': noise_floor,
        'responsivity': responsivity,
        'signal_response': signal_response
    }





def compare_detection_thresholds(p_data, kgrid, source_pressure, snr_db=10, detection_sigma=5):
    """
    Properly compares detection thresholds before/after matched filtering
    with correct SNR calculation in matched filter domain.
    """
    # 1. Create realistic noisy signal
    clean_signal = np.sum(p_data, axis=tuple(range(1, p_data.ndim)))
    noisy_signal = add_noise_fixed_sigma(clean_signal, sigma=100)
    
    # 2. Calculate raw detection threshold (time domain)

    
    # 3. Apply matched filter
    mf_result = apply_match_filter(p_data, kgrid, 100)
    y = mf_result['matched_output']  # Correlation-domain output

    raw_noise_floor = np.std(noisy_signal - clean_signal)
    raw_snr = mf_result['raw_detection_snr']
    raw_threshold = (source_pressure / raw_snr) * detection_sigma
    
    # 4. Calculate filtered SNR properly
    # Find main lobe
    peak_idx = np.argmax(np.abs(y))
    main_lobe_width = len(mf_result['template'])  # Width of our matched filter
    
    # Create noise measurement mask (exclude main lobe + guard band)
    guard_band = main_lobe_width // 2
    noise_mask = np.ones(len(y), dtype=bool)
    noise_mask[max(0, peak_idx-guard_band-main_lobe_width):min(len(y), peak_idx+guard_band+main_lobe_width)] = False
    
    # Verify we have enough noise samples
    if np.sum(noise_mask) < 100:
        noise_mask = np.ones(len(y), dtype=bool)
        noise_mask[peak_idx-50:peak_idx+50] = False
    
    filtered_noise_floor = np.std(y[noise_mask])
    filtered_snr = mf_result['mf_detection_snr']
    filtered_threshold = (source_pressure / filtered_snr) * detection_sigma
    
    # 5. Verify results make physical sense
    if filtered_noise_floor <= 0:
        raise ValueError("Invalid noise floor measurement after matched filtering")
    
    return {
        'raw_threshold_pa': raw_threshold,
        'filtered_threshold_pa': filtered_threshold,
        'improvement_factor': raw_threshold / filtered_threshold,
        'raw_snr': raw_snr,
        'filtered_snr': filtered_snr,
        'main_lobe_width': main_lobe_width  # For debugging
    }