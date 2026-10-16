
# DEDECHAO

DEDECHAO is a Python framework for generating spatial and temporal fractal and multifractal fields based on Gaussian Multiplicative Chaos (GMC), with temporal dynamics driven by Ornstein–Uhlenbeck processes. It is particularly suited to modeling intermittent isotropic turbulence.

## Features

- Generation of fractal and multifractal fields in 2D and 3D, with temporal dynamics.
- Fourier and Wavelet resolution available.
- Spatial and temporal metrics for field evaluation.
- Video generation for time-dependent fields.
- Physically constrained parameter selection to promote realistic fields.

## Usage

Run `Exec` to launch the simulations. Simulation parameters are defined in `sim_config`.

- `Bi_OU_core`: main implementation.
- `Bi_OU_core_Wavelet`: wavelet-based implementation.
- `plot`: plotting and visualization functions, called from `Exec`.

## Requirements

Main dependencies include [PyTorch](https://pytorch.org/) and [pytorch_wavelets](https://github.com/fbcotter/pytorch_wavelets).

## Scientific context

The framework combines Gaussian Multiplicative Chaos for spatial multifractal construction with Ornstein–Uhlenbeck processes for temporal dynamics.

## Citation

If you use DEDECHAO in your research, please cite this repository and any associated publications.

## License

See the `LICENSE` file for terms of use.
