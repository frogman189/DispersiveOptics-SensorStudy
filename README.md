# DispersiveOptics-SensorStudy
This repository provides tools and simulations for evaluating how different optical sensor geometries and types perform when detecting a single red blood cell (modeled as a point source) in a dispersive medium. It’s intended as a compact framework for your “Optics in Dispersive Media & Biomedical Imaging” course project.

Key features:

    Dispersive‐media propagation
    – Implements wavelength‐dependent refractive index models
    – Simulates pulse broadening and distortion through tissue‐like media

    Point‐source RBC model
    – Represents a red blood cell as a time‐modulated point emitter
    – Configurable emission spectrum and temporal profile

    Sensor shape and type comparison
    – Supports planar, curved, annular, and segmented sensor masks
    – Allows choice of photodiode, CCD, and CMOS‐style detection models
    – Quantifies sensitivity, spatial resolution, and signal‐to‐noise metrics

    Visualization & analysis
    – Plots impulse responses, dispersion curves, and noise‐added signals
    – Computes metrics such as peak amplitude shift, pulse width, and detection error

Getting started:

    Clone the repo and install dependencies (requirements.txt)

    Edit config.yaml to choose your medium dispersion, RBC emission, and sensor parameters

    Run simulate.py to generate propagation and detection data

    Use analyze.py or the Jupyter notebooks in /notebooks to reproduce figures and compare sensor layouts

Feel free to tweak the sensor definitions in sensors.py or add new dispersion models in media.py to explore further!
