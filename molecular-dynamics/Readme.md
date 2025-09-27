# Computational Materials Science: Molecular Dynamics

This repository contains Python implementations of key molecular dynamics algorithms and concepts. The code provides a foundation for atomic-scale simulations of materials, focusing on fundamental algorithms in computational materials science.

## Overview

Molecular Dynamics (MD) is a computational technique that uses numerical simulations to study the time evolution of molecular, atomic, or particle systems under classical mechanics. It allows for the observation and analysis of the dynamic behavior of molecular systems at the microscopic scale, establishing relationships between the microscopic and macroscopic aspects of the system through statistical physics.

## Key Components Implemented

### Core Components

- **Particle System Implementation**: Classes to represent particles, simulation boxes, and boundary conditions
- **Periodic Boundary Conditions**: Implementation of periodic boundary conditions for simulating infinite systems
- **Neighbor List Algorithms**: Efficient computation of interatomic interactions using Verlet lists
- **Interatomic Potentials**: Implementation of the Lennard-Jones potential with various cutoff schemes

### Algorithms

1. **Integration Algorithms**
   - Forward Euler Method
   - Position Verlet Algorithm
   - Velocity Verlet Algorithm 
   - Leapfrog Algorithm

2. **Optimization Algorithms**
   - Steepest Descent Method
   - Conjugate Gradient Method
   - Backtracking Line Search


## Notebooks

The repository includes several Jupyter notebooks that demonstrate the algorithms and provide educational content:

1. `computational_materials_science-molecular_dynamics-algorithm_principle.ipynb`: Explains the theoretical foundations
2. `computational_materials_science-molecular_dynamics-LAMMPS_examples.ipynb`: Provides examples using the LAMMPS software
3. `integration_algorithms_for_molecular_dynamics_python.ipynb`: Details the integration algorithms
4. `Lennard-Jones_potential_python.ipynb`: Implements the Lennard-Jones potential
5. `optimization_algorithms_for_molecular_dynamics_python.ipynb`: Details the optimization methods
6. `syntax_and_algorithms_python.ipynb`: Reviews Python programming concepts for MD

## References

The implementation is based on principles described in:

1. Frenkel, D. and Smit, B. "Understanding Molecular Simulation: From Algorithms to Applications"
2. Allen, M. P. and Tildesley, D. J. "Computer Simulation of Liquids"
3. Tuckerman, M. E. "Statistical Mechanics: Theory and Molecular Simulation"

## License

This project is licensed under the CC BY-NC-SA 4.0 International License - see individual notebooks for details.

## Author

Wang Jianghai - [📧Email Contact](mailto:jianghai001@e.ntu.edu.sg)
