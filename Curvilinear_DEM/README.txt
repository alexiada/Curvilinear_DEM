CURVILINEAR PARTICLE DYNAMICS FOR DEM AND PARTICLE-BASED METHODS

This repository contains the Python code and Jupyter notebooks developed for the paper:

Curvilinear Particle Dynamics for DEM and Particle-Based Methods: Simplifying Complex Boundaries with Coordinate Maps

Alessio Alexiadis

The code demonstrates a curvilinear formulation for the Discrete Element Method (DEM) and related particle-based methods. Particle positions and velocities are advanced in a mapped coordinate system in which a complex physical boundary becomes a simple fixed coordinate surface. Physical coordinates are retained for operations such as particle-particle contact, neighbour searches, visualisation, and analysis.

The repository includes single-particle and multiparticle validation against Cartesian reference simulations, followed by four case studies:

1. Rotating drums with irregular Fourier-generated boundaries.
2. Granular flow in a rough trapezium using a hybrid Cartesian-curvilinear solver.
3. A deforming boundary represented by a time-dependent coordinate map.
4. A three-dimensional irregular container with rotating gravity.

The original particle-simulation code was developed for:

A. Alexiadis, "A minimalistic approach to physics-informed machine learning using neighbour lists as physics-optimized convolutions for inverse problems involving particle systems," Journal of Computational Physics, 473, 111750 (2023).

https://doi.org/10.1016/j.jcp.2022.111750

For the present study, the original code was substantially adapted, extended, optimized, and refactored. OpenAI Codex was used as a software-development assistant for optimization, refactoring, and parts of the implementation workload. The scientific formulation, modelling decisions, testing, interpretation, and final responsibility for the code remain with the author.

REQUIREMENTS AND USE

The code uses Python, NumPy, Numba, Matplotlib, Jupyter, and IPython. These packages can be installed with:

python -m pip install numpy numba matplotlib jupyter ipython

JupyterLab can then be started with:

jupyter lab

Open the required notebook and run its cells in order. Keep the directory structure unchanged because the notebooks import Python modules from their own folders. The first execution of a Numba-compiled function may be slower because it is compiled at run time.

This is research code intended to support reproducibility and further investigation. It is not a general-purpose or safety-certified DEM package.

CITATION

If this code contributes to a publication, report, thesis, presentation, or derived software, please acknowledge the source and cite the associated paper:

Alexiadis, A., "Curvilinear Particle Dynamics for DEM and Particle-Based Methods: Simplifying Complex Boundaries with Coordinate Maps."

Please add the final journal, year, volume, article number, and DOI when available.

LICENSE

Copyright (C) 2026 Alessio Alexiadis.

This software is free software: you may redistribute it and/or modify it under the terms of the GNU General Public License, version 3, as published by the Free Software Foundation.

The GNU General Public License version 3 permits use for both non-commercial and commercial purposes. If a modified version is distributed, the corresponding source code must also be made available under GPLv3. Please retain the copyright notice and acknowledge the source in academic and scientific work.

This software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

The complete GNU GPLv3 terms are available at:

https://www.gnu.org/licenses/gpl-3.0.html
