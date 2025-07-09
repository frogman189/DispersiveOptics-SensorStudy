import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import correlate
from scipy.signal import correlate, correlation_lags


def add_gaussian_noise(signal: np.ndarray,
                       snr_db: float = 20) -> dict:
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



def plot_presure_and_noise_presure_over_time(p, kgrid, snr_db=20):
    """
    Plot the summed raw pressure and noise-added pressure over time, side by side.

    Args:
        p (ndarray): Array of shape (Nt, ...) containing raw pressures.
        kgrid: Object with attribute t_array of shape (1, Nt) or (Nt,).
        snr_db (float): Desired signal-to-noise ratio in dB for noise addition.
    """
    # Add Gaussian noise to p

    noise_p = add_gaussian_noise(p, snr_db=snr_db)

    # Sum over spatial axes to get time series (shape: Nt)
    p_total         = np.sum(p,      axis=tuple(range(1, p.ndim)))
    p_total_noisy   = np.sum(noise_p, axis=tuple(range(1, noise_p.ndim)))

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
    plt.show()



def apply_match_filter(p: np.ndarray, kgrid, snr_db: float = 20, win_samples: int = 100) -> dict:
    """
    Apply a matched filter to a spatial pulse to detect its arrival time, with detailed visualization.

    Steps:
      1. Collapse the spatial data into a 1-D "clean" pulse by summing over all non-time dimensions.
      2. Add Gaussian noise at the specified SNR (dB) and compute the pure noise component.
      3. Locate the main pulse in the clean data and extract a window around it as the template.
      4. Zero-mean and unit-energy normalize the template.
      5. Zero-center the received noisy signal and compute full cross-correlation.
      6. Build the lag axis in seconds and keep only causal lags (≥ 0).
      7. Detect the arrival time by finding the peak in the matched-filter output.
      8. Plot a 2×2 grid showing noise-only, clean pulse, noisy vs matched output, and clean vs matched output.

    Args:
        p (np.ndarray): Multi-dimensional pulse data (time × spatial dims).
        kgrid: Object with `t_array` attribute (shape (1, Nt) or (Nt, 1)) giving time vector (s).
        snr_db (float): Desired signal-to-noise ratio, in dB.
        win_samples (int): Number of samples for the template window around the pulse peak.

    Returns:
        dict: {
            'noise_only': np.ndarray, raw noise component (p_noisy - p_total), shape (Nt,);
            'clean_pulse': np.ndarray, spatially summed pulse before noise, zero-mean, shape (Nt,);
            'received': np.ndarray, zero-mean noisy signal, shape (Nt,);
            'lags': np.ndarray, causal lag times for correlation (s), shape (N_lags,);
            'matched_output': np.ndarray, normalized matched-filter output, shape (N_lags,);
            'detection_time': float, estimated pulse arrival time (s).
        }
    """
    # 1) Collapse spatially into 1-D clean pulse
    p_total = np.sum(p, axis=tuple(range(1, p.ndim)))
    t = kgrid.t_array.flatten()
    dt = t[1] - t[0]

    # 2) Add noise and compute pure noise
    p_noisy = add_gaussian_noise(p_total, snr_db=snr_db)
    noise_only = p_noisy - p_total

    # 3) Locate main pulse and extract template window
    peak_idx_clean = np.argmax(np.abs(p_total))
    i0 = max(0, peak_idx_clean - win_samples // 2)
    i1 = min(len(p_total), peak_idx_clean + win_samples // 2)
    template = p_total[i0:i1].copy()
    if template.size == 0 or np.allclose(template, 0):
        raise RuntimeError(f"Template window [{i0}:{i1}] is empty or zero.")

    # 4) Zero-mean & unit-energy normalize the template
    template -= np.mean(template)
    E = np.dot(template, template)
    template /= np.sqrt(E)

    # 5) Prepare received signal and cross-correlate
    received = p_noisy - np.mean(p_noisy)
    corr_full = correlate(received, template, mode='full')

    # 6) Build lag axis and keep causal part
    lags = correlation_lags(len(received), len(template), mode='full') * dt
    pos = lags >= 0
    lags_pos = lags[pos]
    corr_pos = corr_full[pos]

    # 7) Find detection peak
    peak_idx = np.argmax(corr_pos)
    t_detect = lags_pos[peak_idx]

    # 8) Return structured outputs
    return {
        'noise_only':      noise_only,
        'clean_pulse':     p_total,
        'received':        received,
        'lags':            lags_pos,
        'matched_output':  corr_pos,
        'detection_time':  t_detect
    }


def plot_match_filter_results(results: dict,
                            t: np.ndarray,
                            snr_db: float):
    
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
