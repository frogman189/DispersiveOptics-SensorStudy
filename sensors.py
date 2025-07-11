import numpy as np
from kwave.ksensor import kSensor
import matplotlib.pyplot as plt
import os


def _create_point_sensor(sim_params, **kwargs):
    """
    Point sensor at the source position.
    """
    kgrid = sim_params['kgrid']
    Nx, Ny, Nz = kgrid.Nx, kgrid.Ny, kgrid.Nz
    source_pos = sim_params['source_pos']

    sensor = kSensor()
    sensor.record = ['p']
    sensor.mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    sensor.mask[source_pos[0], source_pos[1], Nz - 1] = True

    sens = np.zeros((Nx, Ny), dtype=float)
    sens[source_pos[0], source_pos[1]] = 1.0
    sensor.sensitivity_map = sens

    print("Point sensor created!")

    return sensor


def _create_linear_delta_sensor(sim_params, **kwargs):
    """
    Linear delta sensor along x-axis centered at source.
    Only the central point of the defined length is active; sensitivity=1.
    Optional kwargs:
      length (meters): physical length of the array (default 1e-3)
    """
    kgrid = sim_params['kgrid']
    Nx, Ny, Nz = kgrid.Nx, kgrid.Ny, kgrid.Nz
    dx = sim_params['dx']
    source_pos = sim_params['source_pos']
    length = kwargs.get('length', 1e-3)
    pts = max(1, round(length / dx))

    # Compute the starting x and central index
    start_x = source_pos[0] - pts // 2
    center_idx = start_x + pts // 2
    # Ensure indices are within bounds
    center_idx = int(np.clip(center_idx, 0, Nx - 1))
    center_y = int(np.clip(source_pos[1], 0, Ny - 1))

    sensor = kSensor()
    sensor.record = ['p']
    # Mask: only the central point is active
    sensor.mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    #sensor.mask[center_idx, center_y] = True
    for i in range(pts):
        sensor.mask[start_x + i, source_pos[1], Nz - 1] = True

    # Sensitivity map: 1 at the central point, 0 elsewhere
    sens = np.zeros((Nx, Ny), dtype=float)
    sens[center_idx, center_y] = 1.0
    sensor.sensitivity_map = sens
        
    print("Linear delta sensor created!")

    return sensor


def _create_linear_gaussian_sensor(sim_params, **kwargs):
    """
    Linear sensor with Gaussian sensitivity along x-axis.
    Optional kwargs: length (meters), sigma (meters)
    """
    kgrid = sim_params['kgrid']
    Nx, Ny, Nz = kgrid.Nx, kgrid.Ny, kgrid.Nz
    dx = sim_params['dx']
    source_pos = sim_params['source_pos']
    length = kwargs.get('length', 1e-3)
    sigma = kwargs.get('sigma', length/4)
    pts = round(length / dx)
    start_x = source_pos[0] - pts // 2

    sensor = kSensor()
    sensor.record = ['p']
    sensor.mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    sensitivity = np.zeros((Nx, Ny))

    x = np.linspace(-length/2, length/2, pts)
    weights = np.exp(-x**2 / (2*sigma**2))
    weights /= np.sum(weights)

    for i in range(pts):
        px = start_x + i
        py = source_pos[1]
        sensor.mask[px, py, Nz - 1] = True
        sensitivity[px, py] = weights[i]

    sensor.sensitivity_map = sensitivity

    print("Linear gaussian sensor created!")

    return sensor


def _create_linear_uniform_sensor(sim_params, **kwargs):
    """
    Linear sensor with uniform sensitivity along x-axis.
    Optional kwargs: length (meters)
    """
    kgrid = sim_params['kgrid']
    Nx, Ny, Nz = kgrid.Nx, kgrid.Ny, kgrid.Nz
    dx = sim_params['dx']
    source_pos = sim_params['source_pos']
    length = kwargs.get('length', 1e-3)
    pts = round(length / dx)
    start_x = source_pos[0] - pts // 2

    sensor = kSensor()
    sensor.record = ['p']
    sensor.mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    sensitivity = np.zeros((Nx, Ny))

    weight = 1.0 / pts
    for i in range(pts):
        px = start_x + i
        py = source_pos[1]
        sensor.mask[px, py, Nz - 1] = True
        sensitivity[px, py] = weight

    sensor.sensitivity_map = sensitivity

    print("Linear uniform sensor created!")

    return sensor


def _create_rectangular_gaussian_sensor(sim_params, **kwargs):
    """
    Rectangular gaussian sensor region centered on source.
    Optional kwargs: length (m), width (m), sigma (m)
    """
    kgrid = sim_params['kgrid']
    Nx, Ny, Nz = kgrid.Nx, kgrid.Ny, kgrid.Nz
    dx = sim_params['dx']
    source_pos = sim_params['source_pos']
    length = kwargs.get('length', 1e-3)
    width  = kwargs.get('width', 0.5e-3)
    sigma  = kwargs.get('sigma', length/4)

    lp = round(length / dx)
    wp = round(width  / dx)
    sx = source_pos[0] - lp//2
    sy = source_pos[1] - wp//2

    sensor = kSensor()
    sensor.record = ['p']
    sensor.mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    sensitivity = np.zeros((Nx, Ny))


    xx, yy = np.meshgrid(
        np.linspace(-length/2, length/2, lp),
        np.linspace(-width /2, width /2, wp)
    )
    weights = np.exp(-(xx**2 + yy**2) / (2*sigma**2))
    weights /= np.sum(weights)
    for i in range(lp):
        for j in range(wp):
            px, py = sx + i, sy + j
            sensor.mask[px, py, Nz - 1] = True
            sensitivity[px, py] = weights[j, i]

    sensor.sensitivity_map = sensitivity

    print("Rectangular gaussian sensor created!")

    return sensor


def _create_rectangular_uniform_sensor(sim_params, **kwargs):
    """
    Rectangular uniform sensor region centered on source.
    Optional kwargs: length (m), width (m)
    """
    kgrid = sim_params['kgrid']
    Nx, Ny, Nz = kgrid.Nx, kgrid.Ny, kgrid.Nz
    dx = sim_params['dx']
    source_pos = sim_params['source_pos']
    length = kwargs.get('length', 1e-3)
    width  = kwargs.get('width', 0.5e-3)

    lp = round(length / dx)
    wp = round(width  / dx)
    sx = source_pos[0] - lp//2
    sy = source_pos[1] - wp//2

    sensor = kSensor()
    sensor.record = ['p']
    sensor.mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    sensitivity = np.zeros((Nx, Ny))

    w = 1.0 / (lp * wp)
    for i in range(lp):
        for j in range(wp):
            px, py = sx + i, sy + j
            sensor.mask[px, py, Nz - 1] = True
            sensitivity[px, py] = w
   

    sensor.sensitivity_map = sensitivity

    print("Rectangular uniform sensor created!")
    return sensor

def make_snake_mask(rect_mask: np.ndarray) -> np.ndarray:
    """
    Given a 2D boolean array containing a single rectangular block of True,
    return a new boolean array of the same shape where that rectangle
    has been replaced by a “snake” (full rows alternating with single-edge points).
    """
    if rect_mask.dtype != bool:
        raise ValueError("Input mask must be of boolean type.")
    
    H, W = rect_mask.shape
    
    # Find the bounds of the rectangle:
    rows_any = np.any(rect_mask, axis=1)
    cols_any = np.any(rect_mask, axis=0)
    if not rows_any.any() or not cols_any.any():
        raise ValueError("Input mask contains no True values.")
    
    y_min = np.argmax(rows_any)
    y_max = H - 1 - np.argmax(rows_any[::-1])
    x_min = np.argmax(cols_any)
    x_max = W - 1 - np.argmax(cols_any[::-1])
    
    # Prepare new mask
    snake = np.zeros_like(rect_mask)
    
    # Build the snake
    snake_right = True
    for i, y in enumerate(range(y_min, y_max + 1)):
        if i % 2 == 0:
            # even index → full horizontal segment
            snake[y, x_min:x_max + 1] = True
        else:
            # odd index → one True at the current edge
            if snake_right:
                snake[y, x_max] = True
            else:
                snake[y, x_min] = True
            snake_right = not snake_right
    
    return snake


def _create_snake_sensor(sim_params, **kwargs):
    """
    Snake-shaped sensor defined by a snake path within a rectangular region.
    First creates a rectangular sensor (uniform or gaussian), then applies a snake-shaped
    mask, zeroes out sensitivity outside the snake, and renormalizes.
    Optional kwargs: length (m), width (m), distribution ('uniform'|'gaussian'), sigma (m), segments (int)
    """
    dist = kwargs.get('distribution', 'uniform')

    kgrid = sim_params['kgrid']
    Nx, Ny, Nz = kgrid.Nx, kgrid.Ny, kgrid.Nz
    
    # Build base rectangular sensor for distribution
    if dist == 'uniform':
        base = _create_rectangular_uniform_sensor(sim_params, **kwargs)
    else:
        base = _create_rectangular_gaussian_sensor(sim_params, **kwargs)
    rect_mask = base.mask
    rect_sens = base.sensitivity_map

    rect_mask_slice = rect_mask[ :, :, Nz - 1]
    mask_snake_slice = make_snake_mask(rect_mask_slice)

    # Apply snake mask to sensitivity and renormalize
    sens = rect_sens * mask_snake_slice
    total = sens.sum()
    if total > 0:
        sens = sens / total

    # Build final sensor
    sensor = kSensor()
    sensor.record = ['p']
    sensor.mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    sensor.mask[ :, :, Nz - 1] = mask_snake_slice
    sensor.sensitivity_map = sens
    return sensor


def _create_rectangular_checkerboard_sensor(sim_params, **kwargs):
    """
    Rectangular sensor (uniform or gaussian) with checkerboard masking.
    Optional kwargs: length (m), width (m), distribution ('uniform'|'gaussian'), sigma (m)
    """
    dist   = kwargs.get('distribution', 'uniform')

    # First create a full rectangular sensor
    if dist == 'uniform':
        sensor = _create_rectangular_uniform_sensor(sim_params, **kwargs)
    else:
        sensor = _create_rectangular_gaussian_sensor(sim_params, **kwargs)
    # Apply checkerboard pattern to mask only
    Nx, Ny, Nz = sensor.mask.shape
    # create checkerboard boolean matrix: True on (i+j)%2==0
    checker = ((np.add.outer(np.arange(Nx), np.arange(Ny))) % 2 == 0)
    sensor.mask[:, :, Nz - 1] = sensor.mask[:, :, Nz - 1] & checker
    # sensitivity_map untouched
    return sensor


_TYPE_MAP = {
    1: _create_point_sensor,
    2: _create_linear_delta_sensor,
    3: _create_linear_gaussian_sensor,
    4: _create_linear_uniform_sensor,
    5: _create_rectangular_gaussian_sensor,
    6: _create_rectangular_uniform_sensor,
    7: _create_snake_sensor,
    8: _create_rectangular_checkerboard_sensor,
}


def create_sensor_3D(sensor_type: int, sim_params: dict, **kwargs):
    """
    Factory: create a 3D sensor by type:
      1: point
      2: linear delta
      3: linear Gaussian
      4: linear uniform
      5: rectangular gaussian
      6: ectangular uniform
      7: snake-shaped linear
      8: rectangular + checkerboard
    """
    try:
        creator = _TYPE_MAP[sensor_type]
    except KeyError:
        raise ValueError(f"Unknown sensor_type {sensor_type}")
    sensor = creator(sim_params, **kwargs)
    return sensor


def plot_sensor_sensitivity_and_save(sensor, sim_params, sensor_type: int, output_dir: str):
    """Plot sensor mask and sensitivity side by side in the XY plane, and save to disk.

    Args:
        sensor: kSensor object with .mask and optional .sensitivity_map
        sim_params: dict containing 'kgrid' and optional 'simulation_type'
        sensor_type: int code for sensor geometry (see SENSOR_FACTORY)
        output_dir: base directory where plots will be saved

    Factory mapping for sensor_type:
      1: point
      2: linear delta
      3: linear gaussian
      4: linear uniform
      5: rectangular gaussian
      6: rectangular uniform
      7: snake-shaped linear
      8: rectangular + checkerboard
    """
    # Map sensor_type codes to descriptive names
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
    # Determine folder name
    sensor_name = SENSOR_FACTORY.get(sensor_type, f'type_{sensor_type}')
    save_dir = os.path.join(output_dir, sensor_name)
    os.makedirs(save_dir, exist_ok=True)

    # Extract grid dims
    kgrid = sim_params['kgrid']
    Nx, Ny, Nz = kgrid.Nx, kgrid.Ny, kgrid.Nz

    # Determine 2D mask and sensitivity
    if sim_params.get('simulation_type') == '3D':
        mask_2d = sensor.mask[:, :, Nz - 1]
        sens_map = getattr(sensor, 'sensitivity_map', None)
        # If a full 3D map is provided, assume last z-slice
        sens_2d = sens_map[:, :, Nz - 1] if sens_map is not None else None
    else:
        mask_2d = sensor.mask
        sens_map = getattr(sensor, 'sensitivity_map', None)
        sens_2d = sens_map if sens_map is not None else None

    # Fallback to mask if no sensitivity_map
    sens_plot = sens_2d if sens_2d is not None else mask_2d.astype(float)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    ax1.imshow(mask_2d.T, origin='lower', cmap='gray', vmin=0, vmax=1)
    ax1.set_title('Sensor Mask')
    ax1.set_xlabel('X Position [grid points]')
    ax1.set_ylabel('Y Position [grid points]')

    im = ax2.imshow(sens_plot.T, origin='lower', cmap='viridis')
    ax2.set_title('Sensitivity Map')
    ax2.set_xlabel('X Position [grid points]')
    ax2.set_ylabel('')
    fig.colorbar(im, ax=ax2, label='Sensitivity')

    plt.tight_layout()

    # Save to file instead of showing
    save_path = os.path.join(save_dir, 'sensor.png')
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"Saved sensor plot to: {save_path}")


def plot_sensor_sensitivity(sensor, sim_params):
    """Plot sensor mask and sensitivity side by side in the XY plane."""

    kgrid = sim_params['kgrid']
    Nx, Ny, Nz = kgrid.Nx, kgrid.Ny, kgrid.Nz

    # Determine 2D mask and sensitivity based on simulation dimension
    if sim_params.get('simulation_type') == '3D':
        mask_2d = sensor.mask[:, :, Nz - 1]
        sens_map = getattr(sensor, 'sensitivity_map', None)
        sens_2d = sens_map if sens_map is not None else None
        #sens_2d = sens_map[:, :, z_idx] if sens_map is not None else None
    else:
        mask_2d = sensor.mask
        sens_map = getattr(sensor, 'sensitivity_map', None)
        sens_2d = sens_map if sens_map is not None else None

    # Prepare sensitivity for plotting (fallback to mask if no sensitivity_map)
    if sens_2d is None:
        sens_plot = mask_2d.astype(float)
    else:
        sens_plot = sens_2d

    # Plot side by side
    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(mask_2d.T, origin='lower', cmap='gray', vmin=0, vmax=1)
    ax1.set_title('Sensor Mask')
    ax1.set_xlabel('X Position [grid points]')
    ax1.set_ylabel('Y Position [grid points]')

    ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
    im = ax2.imshow(sens_plot.T, origin='lower', cmap='viridis')
    plt.colorbar(im, ax=ax2, label='Sensitivity')
    ax2.set_title('Sensitivity Map')
    ax2.set_xlabel('X Position [grid points]')
    ax2.set_ylabel('')

    plt.tight_layout()
    plt.show()