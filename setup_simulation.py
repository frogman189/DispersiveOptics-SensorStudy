import numpy as np
import matplotlib.pyplot as plt
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D


def setup_simulation(simulation_type, source_position='center'):
    """
    Set up a 2D/ 3D simulation environment using proper k-Wave classes
    for simulating a red blood cell illuminated by laser light.
    """
    # =========================================================================
    # SIMULATION PARAMETERS
    # =========================================================================
    
    # Physical parameters
    c0 = 1500  # speed of sound in blood [m/s]
    rho0 = 1060  # density of blood [kg/m^3]
    alpha_coeff = 0.2  # absorption coefficient [dB/(MHz^y cm)]
    alpha_power = 1.1  # absorption power law exponent
    
    if simulation_type == '3D':

        Nx = Ny = 50
        Nz = 3
        dx = 43e-6 # 43 [um]
        dy = 43e-6 # 43 [um]
        dz = 43e-6 # 43 [um]

        x_grid_size = Nx * dx
        y_grid_size = Ny * dy
        z_grid_size = (Nz - 1) * dz

    else:
        # Grid parameters
        grid_size = 40e-3  # 2mm cube
        N = 50
        dx = grid_size / N
        
    
    # =========================================================================
    # CREATE GRID
    # =========================================================================
    if simulation_type == '3D':
        kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dy, dz])
        t_xy = np.sqrt(2) * x_grid_size / c0            # X-Y diagonal time
        t_z = z_grid_size / c0                          # Z-axis time
        t_end = 1.2 * (t_xy + t_z)                      # Total time with buffer
        kgrid.makeTime(c0, t_end=t_end)
    else:
        kgrid = kWaveGrid([N, N], [dx, dx])
        t_end = 1.2 * np.sqrt(2) * grid_size / c0       # time to cross grid diagonally
        kgrid.makeTime(c0, t_end=t_end)
    
    print("t end:", t_end)
        
    
    # =========================================================================
    # CREATE MEDIUM
    # =========================================================================
    medium = kWaveMedium(
        sound_speed=c0,
        density=rho0,
        alpha_coeff=alpha_coeff,
        alpha_power=alpha_power,
        BonA=0  # No nonlinearity for this simulation
    )
    
    # =========================================================================
    # CREATE SOURCE (Red Blood Cell)
    # =========================================================================
    source = kSource()
    
    # Initialize p0 as a numpy array
    if simulation_type == '3D':
        p0 = np.zeros((Nx, Ny, Nz), dtype=np.float32)
        if source_position == 'center':
            source_pos = [Nx//2, Ny//2, 0]
            p0[Nx//2, Ny//2, 0] = 1e6
        elif source_position == 'lateral_offset':
            offset_y = Ny//4
            source_pos = [Nx//2, Ny//2 + offset_y, 0]
            p0[Nx//2, Ny//2 + offset_y, 0] = 1e6
        else: #else Diagonal Offset
            offset_x = Nx//4
            offset_y = Ny//4
            source_pos = [Nx//2 + offset_x, Ny//2 + offset_y, 0]
            p0[Nx//2 + offset_x, Ny//2 + offset_y, 0] = 1e6

            
    else:
        p0 = np.zeros((N, N), dtype=np.float32)
        source_pos = [N//2, N//2]
        p0[N//2, N//2] = 1e6 # Point source

    source.p0 = p0
    
    # =========================================================================
    # SIMULATION OPTIONS
    # =========================================================================
    # Create properly structured options objects

    pml_size=10
    simulation_options = SimulationOptions(
        pml_size=pml_size,             # Perfectly Matched Layer size [grid points]
        pml_alpha=2,             # PML absorption [Nepers/grid point]
        pml_inside=False,        # PML outside the grid
        data_cast='single',      # Use single precision
        save_to_disk=True,      # Don't save to disk
        use_kspace=True,         # Use k-space correction
        smooth_c0=False,         # Don't smooth sound speed
        smooth_rho0=False,       # Don't smooth density
        smooth_p0=False,         # Don't smooth initial pressure
        save_to_disk_exit=False  # Don't exit after saving
    )
    
    execution_options = SimulationExecutionOptions(
        is_gpu_simulation=False,     # CPU simulation
        device_num=0,                # Use first GPU (adjust if multi-GPU)
        delete_data=False,          # Keep temporary files
        verbose_level=2             # Show progress
    )
    
    # =========================================================================
    # RETURN SIMULATION PARAMETERS
    # =========================================================================
    return {
        'kgrid': kgrid,
        'medium': medium,
        'source': source,
        'simulation_options': simulation_options,
        'execution_options': execution_options,
        'source_pos': source_pos,
        'dx': dx,
        'dy': dy,
        'dz': dz,
        'simulation_type': simulation_type,
        'pml_size': pml_size,
        'sensor_plane': pml_size + Nz -1
    }


def run_simulation(sim_params, sensor):
    """
    Run simulation and apply sensitivity to results.
    
    Args:
        sim_params: from setup_simulation()
        sensor: kSensor object with sensitivity_map
        
    Returns:
        sensor_data: Sensitivity-adjusted recorded data
    """
    # Run simulation (sensor.mask is used during simulation)
    raw_data = kspaceFirstOrder3D(
        kgrid=sim_params['kgrid'],
        medium=sim_params['medium'],
        source=sim_params['source'],
        sensor=sensor,
        simulation_options=sim_params['simulation_options'],
        execution_options=sim_params['execution_options']
    )
    
    return raw_data


def visualize_source(sim_params):
    """Visualize the initial pressure distribution (p0)"""
    p0 = sim_params['source'].p0

    # Create figure
    fig = plt.figure(figsize=(12, 5))

    if sim_params['simulation_type'] == '3D':
        # ... your existing 3D code ...
        ax1 = fig.add_subplot(131)
        ax1.imshow(p0[:, :, sim_params['source_pos'][2]], cmap='hot')
        ax1.set_title('XY Plane')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')

        ax2 = fig.add_subplot(132)
        ax2.imshow(p0[:, sim_params['source_pos'][1], :], cmap='hot')
        ax2.set_title('XZ Plane')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')

        ax3 = fig.add_subplot(133)
        ax3.imshow(p0[sim_params['source_pos'][0], :, :], cmap='hot')
        ax3.set_title('YZ Plane')
        ax3.set_xlabel('Y')
        ax3.set_ylabel('Z')

        plt.suptitle('Initial Pressure Distribution (p0)')
        plt.tight_layout()
        plt.show()

    else:
        # Fix: create an Axes on the Figure and call imshow there
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(p0, cmap='hot')
        ax.set_title('Initial Pressure Distribution (p0)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.tight_layout()
        plt.show()