# Cryo-EM Ab Initio Reconstruction of Molecules with Tetrahedral and Octahedral Symmetry 

## Overview

This project provides code for ab-initio 3D reconstruction of molecular structures from cryo-electron microscopy (cryo-EM) data with T/O symmetry. The code allows users to specify key parameters for reconstruction, including symmetry type, resolution settings, and optimization parameters.

I recommend first running 'pipeline_T_abinitio.py' for a demo of the algorithm on simulated projection images of the 'T' symmetric molecule EMD-10835. Note that this script is interactive and if run from the command line, matplotlib may block the execution of the code when showing projection-images.
It is possible to set simulation parameters such as:

    img_size – Desired resolution to downsample to.
    num_imgs – Number of projection-images to use.
    noise_variance – Control the noise added to the simulated projections.
As well as setting the script to be non-interactive by setting show_projections = False.

This project provides a command-line interface for running the algorithm.


## Prerequisites

Python 3.x

Dependencies specified in requirements.txt

(Optional) Download these .pkl files and add them to the folder to save time when initially running the project. The program searches for these files and if not found creates them and saves for future use.
[Cache files link](https://drive.google.com/drive/folders/1tIeMKDZIsrmYuxMR4t_vRwOqUtnYGcp4?usp=sharing)

## Usage

Run the script using the following command:

python run_ab_initio_TO.py --sym <SYMMETRY> --instack <INPUT_STACK> --outvol <OUTPUT_VOLUME> [OPTIONS]

### Required Arguments

--sym <SYMMETRY>: The symmetry type of the molecule: T/O.

--instack <INPUT_STACK>: Path to the mrc file of the projections.

--outvol <OUTPUT_VOLUME>: Path to save the output volume file.

### Optional Arguments

--cache_file_name <CACHE_FILE>: Name of the cache file (default: None).

--n_theta <N>: Radial resolution (default: 360).

--rotation_resolution <N>: Rotation resolution (default: 150).

--n_r_perc <PERCENTAGE>: Radial percentage (default: 50).

--viewing_direction <THRESHOLD>: Viewing direction threshold (default: 0.996).

--in_plane_rotation <THRESHOLD>: In-plane rotation threshold (default: 5).

--cg_max_iterations <N>: Maximum iterations for conjugate gradient optimization (default: 50).

### Example Usage

python run_ab_initio_TO.py --sym T --instack projections.mrc --outvol output_volume.mrc \
                 --n_theta 180 --rotation_resolution 100 --cg_max_iterations 30

## Notes

Projections with shifts are not supported.

## Author
Itamar Tzafrir


