import numpy as np
import datetime
import os

from setup_simulation import setup_simulation, run_simulation
from sensors import create_sensor_3D, plot_sensor_sensitivity_and_save
from post_processing import process_sensor_data, analyze_sensor_performance
from match_filter import compare_detection_thresholds, plot_presure_and_noise_presure_over_time, apply_match_filter, plot_match_filter_results, calculate_detector_metrics
from comparison import compare_sensor_signals


def save_experiment_summary(output_dir: str,  
                          sensor_params: dict,
                          source_position,
                          performance_metrics: dict,
                          detector_metrics: dict,
                          detection_comparison: dict,
                          signals_comparations: dict):
    """
    Save experiment setup and results to a summary text file, including complete matched filter comparison.
    
    Args:
        output_dir: Directory where to save the summary file
        sensor_params: Dictionary containing sensor parameters including 'sensor_type'
        performance_metrics: Output from analyze_sensor_performance()
        detector_metrics: Output from calculate_detector_metrics()
        detection_comparison: Output from compare_detection_performance() (contains raw_metrics, filtered_metrics, improvement)
    """
    # Sensor type mapping
    SENSOR_TYPES = {
        1: 'point',
        2: 'linear delta',
        3: 'linear gaussian',
        4: 'linear uniform',
        5: 'rectangular gaussian',
        6: 'rectangular uniform',
        7: 'snake-shaped linear gaussian',
        8: 'snake-shaped linear uniform',
        9: 'rectangular + checkerboard gaussian',
        10: 'rectangular + checkerboard uniform'
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamped filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(output_dir, f"experiment_summary_{timestamp}.txt")
    
    # Prepare sensor description
    sensor_name = SENSOR_TYPES.get(sensor_params['sensor_type'], f"unknown_type_{sensor_params['sensor_type']}")
    sensor_description = f"Sensor Type: {sensor_name}"
    
    if sensor_params['sensor_type'] not in [1, 2]:
        # Add detailed parameters for complex sensors
        distribution = 'gaussian' if 'gaussian' in sensor_name.lower() else 'uniform'
        params = []
        for param in ['length', 'width', 'sigma', 'row_spacing']:
            if param in sensor_params:
                params.append(f"{param}={sensor_params[param]}")

        if 'row_spacing' in sensor_params and sensor_params['sensor_type'] in [7, 8]:
            params.append(f"row_spacing={sensor_params['row_spacing']}")
        
        sensor_description += f"\nDistribution: {distribution}\nParameters: {', '.join(params)}"

    # Prepare performance metrics
    perf_metrics_str = "\n=== SENSOR PERFORMANCE METRICS ===\n"
    for k, v in performance_metrics['aggregate_metrics'].items():
        perf_metrics_str += f"{k.replace('_', ' ').title():>30}: {v:.4e}\n"

    source_position_str = "\n=== SOURCE CHARACTERISTICS ===\n"
    source_position_str += f"Source position: {source_position}\n"
    
    # # Prepare detector metrics
    detector_metrics_str = "\n=== DETECTOR CHARACTERISTICS ===\n"
    for k, v in detector_metrics.items():
        display_name = 'D* (Jones)' if k == 'D_star' else k.replace('_', ' ').title()
        detector_metrics_str += f"{display_name:>30}: {v:.4e}\n"
    
    # Prepare detection comparison if available
    detection_str = ""
    detection_str = "\n=== DETECTION THRESHOLD ANALYSIS ===\n"
    
    # Raw metrics section
    detection_str += "-"*50 + "\n"
    detection_str += f"{'Raw detection threshold (5σ) (Pa)':>30}: {detection_comparison['raw_threshold_pa']:.4e}\n"
    detection_str += f"{'Filtered threshold':>30}: {detection_comparison['filtered_threshold_pa']:.4e}\n"
    detection_str += f"{'SNR improved from':>30} {detection_comparison['raw_snr']:.1f} to {detection_comparison['filtered_snr']:.1f}\n"
    detection_str += f"{'Effective improvement':>30}: {detection_comparison['improvement_factor']:.1f}x\n"

    # write peak SNRs in dB if provided
    comparation_str = ""
    comparation_str = "\n=== SIGNAL AND AUTOCORRELATION ANALYSIS ===\n"

    comparation_str += f"{'Center peak SNR (dB)':>30}: {signals_comparations['center_snr_db']:.1f}\n"
    comparation_str += f"{'Edge peak SNR (dB)':>30}: {signals_comparations['edge_snr_db']:.1f}\n"
    comparation_str += f"{'Center autocorr width (us)':>30}: {signals_comparations['center_autocorr_width_us']:.1f}\n"
    comparation_str += f"{'Edge autocorr width (us)':>30}: {signals_comparations['edge_autocorr_width_us']:.1f}\n"
    comparation_str += f"{'Center PSR (dB)':>30}: {signals_comparations['center_psr_db']:.1f}\n"
    comparation_str += f"{'Edge PSR (dB)':>30}: {signals_comparations['edge_psr_db']:.1f}\n"

    # Write to file
    with open(summary_path, 'w') as f:
        f.write("=== EXPERIMENT SUMMARY ===\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(source_position_str + "\n")
        f.write("=== SENSOR CONFIGURATION ===\n")
        f.write(sensor_description + "\n")
        f.write(perf_metrics_str)
        f.write(detector_metrics_str)
        if detection_comparison:
            f.write(detection_str)
        f.write(comparation_str)
    
    print(f"Experiment summary saved to: {summary_path}")
    return summary_path



def run_full_simulation(sensor_params, source_position, output_dir, snr_db=10):
    # Set up simulation
    sim_params = setup_simulation(simulation_type='3D', source_position=source_position)
    
    # Create sensor with Gaussian sensitivity
    sensor = create_sensor_3D(
        sensor_type=sensor_params['sensor_type'], 
        sim_params=sim_params,
        length=sensor_params['length'],
        width=sensor_params['width'],
        sigma=sensor_params['sigma'],
        row_spacing=sensor_params['row_spacing']
    )

    save_dir = plot_sensor_sensitivity_and_save(sensor, sim_params, sensor_params['sensor_type'], output_dir)

    # Run simulation and apply sensitivity
    sensor_data = run_simulation(sim_params, sensor)


    if sensor_params['sensor_type'] == 1:
        sensor_data['p'] = sensor_data['p'].reshape(-1,1)

    # Extraction sensor recording and applying the sensitivity of the sensor
    sensor_mask_original_size = np.squeeze(sensor.mask[sim_params['pml_size']:-sim_params['pml_size'], sim_params['pml_size']:-sim_params['pml_size'], sim_params['sensor_plane']])
    _, weighted_p, info_dict = process_sensor_data(sensor_mask_original_size, sensor.sensitivity_map, sensor_data['p'])

    performance_metrics = analyze_sensor_performance(
        p_data=sensor_data['p'],
        t_array=sim_params['kgrid'].t_array[0],
        sensor_mask=sensor_mask_original_size,
        sensitivity_map=sensor.sensitivity_map if hasattr(sensor, 'sensitivity_map') else None
    )

    plot_presure_and_noise_presure_over_time(weighted_p, sim_params['kgrid'], snr_db=snr_db, save_to_dir=True, output_dir=save_dir)

    # match_filter_results = apply_match_filter(weighted_p, sim_params['kgrid'], snr_db=snr_db)
    match_filter_results = apply_match_filter(weighted_p, sim_params['kgrid'], sigma_noise=100)
    plot_match_filter_results(match_filter_results, sim_params['kgrid'].t_array.flatten(), snr_db=snr_db, save_to_dir=True, output_dir=save_dir)

    # Calculate fundamental metrics
    if sensor_params['sensor_type'] == 9 or sensor_params['sensor_type'] == 10:
        detector_area = sensor_mask_original_size.size * (sim_params['kgrid'].dx ** 2)
    else:
        detector_area = np.sum(sensor_mask_original_size) * (sim_params['kgrid'].dx ** 2)

    detector_metrics = calculate_detector_metrics(
        p_data=weighted_p,
        noise_data=match_filter_results['noise_only'],
        source_pressure=1e6,
        detector_area=detector_area,
        bandwidth=1/(2*sim_params['kgrid'].dt)  # Nyquist bandwidth
    )

    compare_detection_performance_results = compare_detection_thresholds(
        p_data=weighted_p,
        kgrid=sim_params['kgrid'],
        source_pressure=1e6,
        snr_db=snr_db)
    
    signals_comparations = compare_sensor_signals(info_dict, sensor.sensitivity_map, weighted_p, match_filter_results, sim_params, save_dir)
    
    
    summary_file = save_experiment_summary(
        output_dir=save_dir,
        sensor_params=sensor_params,
        source_position=source_position,
        performance_metrics=performance_metrics,
        detector_metrics=detector_metrics,
        detection_comparison=compare_detection_performance_results,
        signals_comparations=signals_comparations
    )


def main():
    output_dir = '/home/meidanzehavi/electro_optics/DispersiveOptics-SensorStudy/output_results'

    sensor_types_lst = [1, 2, 3, 4, 5, 6, 9, 10]
    length_lst = [1e-3]
    width_lst = [1e-3]
    sigma_lst = [0.3e-3]
    source_position_lst = ['center', 'lateral_offset', 'diagonal_offset']
    row_spacing_snake_lst = [0]

    for sensor_type in sensor_types_lst:
        for length in length_lst:
            for width in width_lst:
                for sigma in sigma_lst:
                    for row_spacing_snake in row_spacing_snake_lst:
                        for source_position in source_position_lst:

                            sensor_params = {'sensor_type': sensor_type,
                                            'length': length,
                                            'width': width,
                                            'sigma': sigma,
                                            'row_spacing': row_spacing_snake}
        
                            run_full_simulation(sensor_params, source_position, output_dir, snr_db=10)

if __name__ == "__main__":
    main()