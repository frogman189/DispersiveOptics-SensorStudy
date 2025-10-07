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


# 06.10.2025 - latest correct
# def apply_match_filter(p: np.ndarray, kgrid, snr_db: float = 10, win_samples: int = 100) -> dict:
#     """
#     Apply a matched filter to a spatial pulse to detect its arrival time, with detailed visualization.
#     Args:
#         p (np.ndarray): Multi-dimensional pulse data (time × spatial dims).
#         kgrid: Object with `t_array` attribute (shape (1, Nt) or (Nt, 1)) giving time vector (s).
#         snr_db (float): Desired signal-to-noise ratio, in dB.
#         win_samples (int): Number of samples for the template window around the pulse peak.

#     Returns:
#         dict: {
#             'noise_only': np.ndarray, raw noise component (p_noisy - p_total), shape (Nt,);
#             'clean_pulse': np.ndarray, spatially summed pulse before noise, zero-mean, shape (Nt,);
#             'received': np.ndarray, zero-mean noisy signal, shape (Nt,);
#             'lags': np.ndarray, causal lag times for correlation (s), shape (N_lags,);
#             'matched_output': np.ndarray, normalized matched-filter output, shape (N_lags,);
#             'detection_time': float, estimated pulse arrival time (s).
#         }
#     """
#     # 1) Collapse spatially into 1-D clean pulse
#     p_total = np.sum(p, axis=tuple(range(1, p.ndim)))
#     t = kgrid.t_array.flatten()
#     dt = t[1] - t[0]

#     # 2) Add noise and compute pure noise
#     #p_noisy = add_gaussian_noise(p_total, snr_db=snr_db)
#     p_noisy = add_noise_fixed_sigma(p_total, sigma=0.1)
#     noise_only = p_noisy - p_total

#     # 3) Locate main pulse and extract template window
#     peak_idx_clean = np.argmax(np.abs(p_total))
#     i0 = max(0, peak_idx_clean - win_samples // 2)
#     i1 = min(len(p_total), peak_idx_clean + win_samples // 2)
#     template = p_total[i0:i1].copy()
#     if template.size == 0 or np.allclose(template, 0):
#         raise RuntimeError(f"Template window [{i0}:{i1}] is empty or zero.")

#     # 4) Zero-mean & unit-energy normalize the template
#     template -= np.mean(template)
#     E = np.dot(template, template)
#     template /= np.sqrt(E)

#     # 5) Prepare received signal and cross-correlate
#     received = p_noisy - np.mean(p_noisy)
#     corr_full = correlate(received, template, mode='full')

#     # 6) Build lag axis and keep causal part
#     lags = correlation_lags(len(received), len(template), mode='full') * dt
#     pos = lags >= 0
#     lags_pos = lags[pos]
#     corr_pos = corr_full[pos]

#     # 7) Find detection peak
#     peak_idx = np.argmax(corr_pos)
#     t_detect = lags_pos[peak_idx]

#     #debug prints
#     # print(f"Peak of clean signal at index: {peak_idx_clean}, time: {t[peak_idx_clean]:.3e} s")
#     # print(f"Template indices: i0={i0}, i1={i1}")
#     # print(f"Template energy: {np.dot(template, template):.3e}")
#     # print(f"Matched filter peak time: {t_detect:.3e}")

#     # 8) Return structured outputs
#     return {
#         'noise_only':      noise_only,
#         'clean_pulse':     p_total,
#         'received':        received,
#         'lags':            lags_pos,
#         'matched_output':  corr_pos,
#         'detection_time':  t_detect,
#         'template':        template
#     }

# def apply_match_filter(p: np.ndarray, kgrid, sigma_noise: float, win_samples: int = 200, remove_dc: bool = False) -> dict:
#     """
#     Apply a matched filter to a spatially-summed pulse and compute SNRs/thresholds
#     on a consistent (energy-based) basis.

#     Key fixes:
#     - Uses a fixed absolute noise std `sigma_noise` for all sensors (fair geometry comparison).
#     - Computes pre-filter (raw) and post-filter SNR **on the same window M** used as the template.
#     - Estimates post-filter noise using only lags with full template overlap (avoids edge variance drop).
#     - Uses a unit-energy template so that post-filter noise variance ≈ sigma_noise^2.

#     Args:
#         p (np.ndarray): Pulse data with shape (Nt, *spatial_dims*).
#         kgrid: Object with `t_array` (shape (Nt,) or (1, Nt) or (Nt, 1)) giving time samples [s].
#         sigma_noise (float): Global noise standard deviation to add (same for all sensors).
#         win_samples (int): Template length M (samples) centered at the clean peak.
#         remove_dc (bool): If True, remove mean from template and received before correlation.

#     Returns:
#         dict with fields:
#             'noise_only': np.ndarray, the noise added (Nt,)
#             'clean_pulse': np.ndarray, spatially-summed clean pulse (Nt,)
#             'received': np.ndarray, noisy received signal after optional DC removal (Nt,)
#             'lags_s': np.ndarray, lag times [s] where full-overlap holds (N_lags_use,)
#             'matched_output': np.ndarray, matched-filter correlation at those lags (N_lags_use,)
#             'detection_time_s': float, estimated arrival time [s] from correlation peak
#             'template': np.ndarray, unit-energy template (M,)
#             'indices': dict, {'i0': int, 'i1': int, 'peak_idx_clean': int}
#             'sigma_raw': float, pre-filter noise std estimate (should ≈ sigma_noise)
#             'sigma_mf': float, post-filter noise std estimate in correlation domain (≈ sigma_noise)
#             'raw_snr_energy': float, (∑ s_win^2) / (M * sigma_raw^2)
#             'mf_snr_energy': float, (y_peak^2) / (sigma_mf^2)
#             'snr_improvement': float, mf_snr_energy / raw_snr_energy
#     """
#     # 1) Collapse spatial dims → 1D clean pulse
#     p_total = np.sum(p, axis=tuple(range(1, p.ndim)))  # shape (Nt,)
#     t = kgrid.t_array.reshape(-1)
#     dt = float(t[1] - t[0])

#     # 2) Add fixed absolute noise (same sigma for every geometry)
#     rng = np.random.default_rng()
#     noise_only = rng.normal(0.0, sigma_noise, size=p_total.shape)
#     p_noisy = p_total + noise_only

#     # 3) Locate main pulse and extract template window of length M=win_samples
#     peak_idx_clean = int(np.argmax(np.abs(p_total)))
#     half = win_samples // 2
#     i0 = max(0, peak_idx_clean - half)
#     i1 = min(len(p_total), i0 + win_samples)
#     # ensure exact length M where possible
#     if (i1 - i0) < win_samples:
#         i0 = max(0, i1 - win_samples)

#     template = p_total[i0:i1].astype(float).copy()
#     if template.size == 0 or np.allclose(template, 0.0):
#         raise RuntimeError(f"Template window [{i0}:{i1}] is empty or zero.")
#     if remove_dc:
#         template -= np.mean(template)
#     E = float(np.dot(template, template))
#     if E <= 0.0:
#         raise RuntimeError("Template has zero energy after DC removal.")
#     template /= np.sqrt(E)  # unit-energy template, so var(noise @ output) ≈ sigma_noise^2

#     # 4) Prepare received signal (optional DC removal)
#     received = p_noisy.astype(float)
#     if remove_dc:
#         received -= np.mean(received)

#     # 5) Correlate (matched filter = cross-correlation with unit-energy template)
#     corr_full = correlate(received, template, mode='full')
#     lags = correlation_lags(len(received), len(template), mode='full')  # in samples

#     # 6) Keep only lags with FULL template overlap (avoids reduced noise variance at edges)
#     M = len(template)
#     # For correlate(x, h) "full", the overlap length varies; full-overlap when 0 <= lag <= len(x)-M
#     mask_full = (lags >= 0) & (lags <= (len(received) - M))
#     corr_use = corr_full[mask_full]
#     lags_use = lags[mask_full]
#     lags_s = lags_use * dt

#     # 7) Peak detection in correlation domain (use absolute value for bipolar pulses)
#     peak_idx_mf = int(np.argmax(np.abs(corr_use)))
#     y_peak = float(np.abs(corr_use[peak_idx_mf]))
#     t_detect = float(lags_s[peak_idx_mf])

#     # # 8) Compute consistent SNRs (energy-based, same window as template)
#     # s_win = p_total[i0:i1].astype(float).copy()
#     # if remove_dc:
#     #     s_win -= np.mean(s_win)

#     # # Pre-filter noise std (should be ~ sigma_noise)
#     # sigma_raw = float(np.std(p_noisy - p_total, ddof=1))
#     # # Post-filter noise std over full-overlap region (should be ~ sigma_noise with unit-energy template)
#     # sigma_mf = float(np.std(corr_use, ddof=1))

#     # raw_snr_energy = float(np.sum(s_win**2)) / (M * (sigma_raw**2) + 1e-12)
#     # mf_snr_energy = (y_peak**2) / (sigma_mf**2 + 1e-12)
#     # snr_improvement = mf_snr_energy / (raw_snr_energy + 1e-12)
    
#     # 8) Compute consistent SNRs
#     s_win = p_total[i0:i1].astype(float).copy()
#     if remove_dc:
#         s_win -= np.mean(s_win)

#     # Pre-filter: peak signal-to-noise ratio
#     peak_signal_raw = float(np.max(np.abs(s_win)))
#     sigma_raw = float(np.std(p_noisy - p_total, ddof=1))
#     raw_snr = (peak_signal_raw**2) / (sigma_raw**2)

#     # Post-filter: estimate noise AWAY from signal peak
#     noise_margin = max(M, 100)  # samples to exclude around peak
#     peak_region_mask = np.abs(lags_use - lags_use[peak_idx_mf]) < noise_margin
#     if np.sum(~peak_region_mask) < 10:
#         # Not enough samples for noise estimation
#         sigma_mf = sigma_raw  # fallback
#     else:
#         sigma_mf = float(np.std(corr_use[~peak_region_mask], ddof=1))

#     mf_snr = (y_peak**2) / (sigma_mf**2)
#     snr_improvement = mf_snr / (raw_snr + 1e-12)

#     return {
#         'noise_only': noise_only,
#         'clean_pulse': p_total,
#         'received': received,
#         'lags': lags_s,
#         'matched_output': corr_use,
#         'detection_time': t_detect,
#         'template': template,
#         'indices': {'i0': i0, 'i1': i1, 'peak_idx_clean': peak_idx_clean, 'peak_idx_mf': peak_idx_mf},
#         'sigma_raw': sigma_raw,
#         'sigma_mf': sigma_mf,
#         'raw_snr_energy': raw_snr,
#         'mf_snr_energy': mf_snr,
#         'snr_improvement': snr_improvement,
#     }


# def apply_match_filter(p: np.ndarray, kgrid, sigma_noise: float, win_samples: int = 200, remove_dc: bool = False) -> dict:
#     """
#     Apply a matched filter to a spatially-summed pulse and compute detection SNRs
#     on a consistent basis.

#     Key approach:
#     - Detection SNR = peak_value / noise_std (dimensionless, in units of σ)
#     - Raw detection: max(|signal|) / sigma_raw
#     - Filtered detection: max(|correlation|) / sigma_mf
#     - Both are comparable because they measure "how many sigmas above noise"

#     Args:
#         p (np.ndarray): Pulse data with shape (Nt, *spatial_dims*).
#         kgrid: Object with `t_array` (shape (Nt,) or (1, Nt) or (Nt, 1)) giving time samples [s].
#         sigma_noise (float): Global noise standard deviation to add (same for all sensors).
#         win_samples (int): Template length M (samples) centered at the clean peak.
#         remove_dc (bool): If True, remove mean from template and received before correlation.

#     Returns:
#         dict with fields:
#             'noise_only': np.ndarray, the noise added (Nt,)
#             'clean_pulse': np.ndarray, spatially-summed clean pulse (Nt,)
#             'received': np.ndarray, noisy received signal after optional DC removal (Nt,)
#             'lags_s': np.ndarray, lag times [s] where full-overlap holds (N_lags_use,)
#             'matched_output': np.ndarray, matched-filter correlation at those lags (N_lags_use,)
#             'detection_time_s': float, estimated arrival time [s] from correlation peak
#             'template': np.ndarray, unit-energy template (M,)
#             'indices': dict, {'i0': int, 'i1': int, 'peak_idx_clean': int, 'peak_idx_mf': int}
#             'sigma_raw': float, pre-filter noise std estimate (should ≈ sigma_noise)
#             'sigma_mf': float, post-filter noise std estimate in correlation domain
#             'raw_detection_snr': float, peak_signal / sigma_raw (dimensionless)
#             'mf_detection_snr': float, y_peak / sigma_mf (dimensionless)
#             'snr_improvement': float, mf_detection_snr / raw_detection_snr
#             'raw_threshold_5sigma': float, detection threshold in Pascals (5 * sigma_raw)
#             'mf_threshold_5sigma': float, detection threshold in correlation units (5 * sigma_mf)
#     """
#     # 1) Collapse spatial dims → 1D clean pulse
#     p_total = np.sum(p, axis=tuple(range(1, p.ndim)))  # shape (Nt,)
#     t = kgrid.t_array.reshape(-1)
#     dt = float(t[1] - t[0])

#     # 2) Add fixed absolute noise (same sigma for every geometry)
#     rng = np.random.default_rng()
#     noise_only = rng.normal(0.0, sigma_noise, size=p_total.shape)
#     p_noisy = p_total + noise_only

#     # 3) Locate main pulse and extract template window of length M=win_samples
#     peak_idx_clean = int(np.argmax(np.abs(p_total)))
#     half = win_samples // 2
#     i0 = max(0, peak_idx_clean - half)
#     i1 = min(len(p_total), i0 + win_samples)
#     # ensure exact length M where possible
#     if (i1 - i0) < win_samples:
#         i0 = max(0, i1 - win_samples)

#     template = p_total[i0:i1].astype(float).copy()
#     if template.size == 0 or np.allclose(template, 0.0):
#         raise RuntimeError(f"Template window [{i0}:{i1}] is empty or zero.")
#     if remove_dc:
#         template -= np.mean(template)
#     E = float(np.dot(template, template))
#     if E <= 0.0:
#         raise RuntimeError("Template has zero energy after DC removal.")
#     template /= np.sqrt(E)  # unit-energy template

#     # 4) Prepare received signal (optional DC removal)
#     received = p_noisy.astype(float)
#     if remove_dc:
#         received -= np.mean(received)

#     # 5) Correlate (matched filter = cross-correlation with unit-energy template)
#     corr_full = correlate(received, template, mode='full')
#     lags = correlation_lags(len(received), len(template), mode='full')  # in samples

#     # 6) Keep only lags with FULL template overlap (avoids reduced noise variance at edges)
#     M = len(template)
#     mask_full = (lags >= 0) & (lags <= (len(received) - M))
#     corr_use = corr_full[mask_full]
#     lags_use = lags[mask_full]
#     lags_s = lags_use * dt

#     # 7) Peak detection in correlation domain (use absolute value for bipolar pulses)
#     peak_idx_mf = int(np.argmax(np.abs(corr_use)))
#     y_peak = float(np.abs(corr_use[peak_idx_mf]))
#     t_detect = float(lags_s[peak_idx_mf])

#     # 8) Compute DETECTION SNRs (peak-to-noise ratio in units of sigma)
    
#     # Raw detection SNR: peak signal in noisy data vs noise std
#     # Use the noisy signal to be fair
#     peak_signal_raw = float(np.max(np.abs(p_noisy[i0:i1])))
    
#     # Estimate noise std from the actual noise (known in simulation)
#     sigma_raw = float(np.std(noise_only, ddof=1))
    
#     # Raw detection SNR (dimensionless: how many sigmas above noise)
#     raw_detection_snr = peak_signal_raw / (sigma_raw + 1e-12)

#     # Post-filter detection SNR: correlation peak vs correlation noise
#     # Estimate noise AWAY from signal peak
#     noise_margin = max(M, 100)  # samples to exclude around peak
#     peak_region_mask = np.abs(lags_use - lags_use[peak_idx_mf]) < noise_margin
    
#     if np.sum(~peak_region_mask) < 10:
#         # Not enough samples for noise estimation - use theoretical value
#         # For unit-energy template: sigma_mf ≈ sigma_raw
#         sigma_mf = sigma_raw
#     else:
#         sigma_mf = float(np.std(corr_use[~peak_region_mask], ddof=1))

#     # Filtered detection SNR (dimensionless: how many sigmas above noise)
#     mf_detection_snr = y_peak / (sigma_mf + 1e-12)
    
#     # SNR improvement factor
#     snr_improvement = mf_detection_snr / (raw_detection_snr + 1e-12)

#     # Detection thresholds (for reference)
#     # These are in DIFFERENT units and should NOT be compared directly!
#     raw_threshold_5sigma = 5.0 * sigma_raw  # in Pascals
#     mf_threshold_5sigma = 5.0 * sigma_mf    # in correlation units

#     return {
#         'noise_only': noise_only,
#         'clean_pulse': p_total,
#         'received': received,
#         'lags': lags_s,
#         'matched_output': corr_use,
#         'detection_time': t_detect,
#         'template': template,
#         'indices': {
#             'i0': i0, 
#             'i1': i1, 
#             'peak_idx_clean': peak_idx_clean, 
#             'peak_idx_mf': peak_idx_mf
#         },
#         'sigma_raw': sigma_raw,
#         'sigma_mf': sigma_mf,
#         'raw_detection_snr': raw_detection_snr,
#         'mf_detection_snr': mf_detection_snr,
#         'snr_improvement': snr_improvement,
#         'raw_threshold_5sigma': raw_threshold_5sigma,
#         'mf_threshold_5sigma': mf_threshold_5sigma,
#     }


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


# def apply_match_filter(p: np.ndarray, kgrid, snr_db: float = 10, win_samples: int = 100) -> dict:
#     """
#     Apply a matched filter to detect pulse arrival time from noisy multidimensional signal.

#     Args:
#         p (np.ndarray): Input signal, shape (Nt, ...) where time is the first dimension.
#         kgrid: Object with 't_array' attribute (shape (Nt,) or (1, Nt)) giving time vector in seconds.
#         snr_db (float): Desired signal-to-noise ratio (dB).
#         win_samples (int): Template window length (samples), centered on pulse peak.

#     Returns:
#         dict with:
#             - 'clean_pulse': 1D spatially summed signal before noise
#             - 'received': Noisy, zero-mean signal
#             - 'noise_only': Noise added to clean signal
#             - 'template': Extracted and normalized matched filter template
#             - 'lags': Time axis (s) of causal matched filter output
#             - 'matched_output': Output of matched filter (normalized)
#             - 'detection_time': Estimated time-of-arrival (s)
#     """
#     # Collapse spatial dimensions to 1D signal
#     p_total = np.sum(p, axis=tuple(range(1, p.ndim)))
#     t = np.ravel(kgrid.t_array)
#     dt = t[1] - t[0]
#     Nt = len(p_total)

#     # Add noise
#     p_noisy = add_gaussian_noise(p_total, snr_db=snr_db)
#     noise_only = p_noisy - p_total

#     # Find main pulse peak
#     peak_idx = np.argmax(np.abs(p_total))
#     half_win = win_samples // 2

#     # Safe template windowing
#     i0 = max(0, peak_idx - half_win)
#     i1 = min(Nt, peak_idx + half_win)
#     template = p_total[i0:i1].copy()

#     if template.size < 5 or np.allclose(template, 0):
#         raise ValueError("Invalid template: too small or all zeros.")

#     # Normalize template: zero-mean, unit energy
#     template -= np.mean(template)
#     template /= np.sqrt(np.sum(template**2) + 1e-12)

#     # Prepare received signal
#     received = p_noisy - np.mean(p_noisy)

#     # Matched filter via cross-correlation (standard approach)
#     corr_full = correlate(received, template, mode='full')
#     lags = correlation_lags(len(received), len(template), mode='full') * dt

#     # Keep causal lags only
#     pos_mask = lags >= 0
#     lags_pos = lags[pos_mask]
#     matched_output = corr_full[pos_mask]

#     # Estimated time of arrival
#     detection_time = lags_pos[np.argmax(matched_output)]

#     return {
#         'clean_pulse':     p_total,
#         'received':        received,
#         'noise_only':      noise_only,
#         'template':        template,
#         'lags':            lags_pos,
#         'matched_output':  matched_output,
#         'detection_time':  detection_time,
#     }


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


# def compare_detection_thresholds(p_data, kgrid, source_pressure, snr_db=10, detection_sigma=5):
#     """
#     Properly compares detection thresholds before/after matched filtering
    
#     Args:
#         p_data: Clean pressure data (without noise)
#         kgrid: Simulation grid
#         source_pressure: Known source amplitude in Pa
#         snr_db: Noise level to add
#         detection_sigma: Detection criterion (e.g., 5σ)
    
#     Returns:
#         dict: Contains meaningful threshold comparison
#     """
#     # 1. Create realistic noisy signal
#     clean_signal = np.sum(p_data, axis=tuple(range(1, p_data.ndim)))
#     noisy_signal = add_gaussian_noise(clean_signal, snr_db)
    
#     # 2. Calculate raw detection threshold
#     raw_noise_floor = np.std(noisy_signal - clean_signal)
#     raw_snr = np.max(clean_signal) / raw_noise_floor
#     raw_threshold = source_pressure / raw_snr * detection_sigma
    
#     # 3. Apply matched filter
#     mf_result = apply_match_filter(p_data, kgrid, snr_db)
    
#     # 4. Calculate filtered detection threshold
#     filtered_noise = mf_result['noise_only']
#     filtered_output = mf_result['matched_output']
    
#     filtered_noise_floor = np.std(filtered_noise)
#     filtered_snr = np.max(filtered_output) / filtered_noise_floor
#     filtered_threshold = source_pressure / filtered_snr * detection_sigma
    
#     # 5. Verify improvement makes physical sense
#     if filtered_threshold >= raw_threshold:
#         raise ValueError("Matched filter should improve detection! Check your implementation")
    
#     return {
#         'raw_threshold_pa': raw_threshold,
#         'filtered_threshold_pa': filtered_threshold,
#         'improvement_factor': raw_threshold / filtered_threshold,
#         'raw_snr': raw_snr,
#         'filtered_snr': filtered_snr
#     }


# def compare_detection_thresholds(p_data, kgrid, source_pressure, snr_db=10, detection_sigma=5):
#     # 1. Create realistic noisy signal
#     clean_signal = np.sum(p_data, axis=tuple(range(1, p_data.ndim)))
#     noisy_signal = add_gaussian_noise(clean_signal, snr_db)
    
#     # 2. Calculate raw detection threshold
#     raw_noise_floor = np.std(noisy_signal - clean_signal)
#     raw_snr = np.max(clean_signal) / raw_noise_floor
#     raw_threshold = source_pressure / raw_snr * detection_sigma
    
#     # 3. Apply matched filter
#     mf_result = apply_match_filter(p_data, kgrid, snr_db)
    
#     # 4. Calculate filtered detection threshold (FIXED)
#     y = mf_result['matched_output']           # correlation-domain output
#     abs_y = np.abs(y)
#     peak_idx = np.argmax(abs_y)

#     # exclude the main lobe with a small guard band
#     guard = max(5, len(y)//100)
#     mask = np.ones_like(y, dtype=bool)
#     mask[max(0, peak_idx-guard):min(len(y), peak_idx+guard+1)] = False

#     filtered_noise_floor = np.std(y[mask]) + 1e-12
#     filtered_snr = abs_y[peak_idx] / filtered_noise_floor
#     filtered_threshold = source_pressure / filtered_snr * detection_sigma
    
#     # # 5. Verify improvement makes physical sense
#     # if filtered_threshold >= raw_threshold:
#     #     raise ValueError("Matched filter should improve detection! Check your implementation")
    
#     return {
#         'raw_threshold_pa': raw_threshold,
#         'filtered_threshold_pa': filtered_threshold,
#         'improvement_factor': raw_threshold / filtered_threshold,
#         'raw_snr': raw_snr,
#         'filtered_snr': filtered_snr
#     }


# def compare_detection_thresholds(p_data, kgrid, source_pressure, snr_db=10, detection_sigma=5):
#     """
#     Properly compares detection thresholds before/after matched filtering
#     with correct SNR calculation in matched filter domain.
#     """
#     # 1. Create realistic noisy signal
#     clean_signal = np.sum(p_data, axis=tuple(range(1, p_data.ndim)))
#     # noisy_signal = add_gaussian_noise(clean_signal, snr_db)
#     noisy_signal = add_noise_fixed_sigma(clean_signal, sigma=100)
    
#     # 2. Calculate raw detection threshold (time domain)
#     raw_noise_floor = np.std(noisy_signal - clean_signal)
#     raw_snr = np.max(np.abs(clean_signal)) / raw_noise_floor
#     #raw_threshold = (source_pressure / raw_snr) * detection_sigma
    
#     # 3. Apply matched filter
#     mf_result = apply_match_filter(p_data, kgrid, 100)
#     raw_threshold = mf_result['raw_detection_snr']
#     y = mf_result['matched_output']  # Correlation-domain output
    
#     # 4. Calculate filtered SNR properly
#     # Find main lobe
#     peak_idx = np.argmax(np.abs(y))
#     main_lobe_width = len(mf_result['template'])  # Width of our matched filter
    
#     # Create noise measurement mask (exclude main lobe + guard band)
#     guard_band = main_lobe_width // 2
#     noise_mask = np.ones(len(y), dtype=bool)
#     noise_mask[max(0, peak_idx-guard_band-main_lobe_width):min(len(y), peak_idx+guard_band+main_lobe_width)] = False
    
#     # Verify we have enough noise samples
#     if np.sum(noise_mask) < 100:
#         noise_mask = np.ones(len(y), dtype=bool)
#         noise_mask[peak_idx-50:peak_idx+50] = False
    
#     filtered_noise_floor = np.std(y[noise_mask])
#     filtered_snr = np.abs(y[peak_idx]) / (filtered_noise_floor + 1e-12)  # Prevent division by zero
#     #filtered_threshold = source_pressure / filtered_snr * detection_sigma
#     filtered_threshold = mf_result['mf_detection_snr']
#     # 5. Verify results make physical sense
#     if filtered_noise_floor <= 0:
#         raise ValueError("Invalid noise floor measurement after matched filtering")
    
#     return {
#         'raw_threshold_pa': raw_threshold,
#         'filtered_threshold_pa': filtered_threshold,
#         'improvement_factor': mf_result['snr_improvement'],
#         'raw_snr': raw_snr,
#         'filtered_snr': filtered_snr,
#         'main_lobe_width': main_lobe_width  # For debugging
#     }



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