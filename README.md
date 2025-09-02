# DNS_Simulation_FeniCSx

Python scripts for 2D incompressible Navier–Stokes simulations using the FeniCSx framework and the Gmsh Python API.

## Description

This repository contains scripts to simulate 2D incompressible flows around obstacles using the FEniCSx finite element framework. The code includes mesh generation, variational formulation, boundary conditions, and post-processing tools for mean flow, forcing, drag, and lift computation.

## Files

- `DNS_V5.py`  
  Main simulation script. Contains:
  - Mesh generation using Gmsh
  - Definition of function spaces (Taylor-Hood P2-P1)
  - Time-stepping solver using the IPCS method
  - Calculation of mean flow, fluctuations, and forcing
  - Drag and lift evaluation
  - Output using VTXWriter and ADIOS4Dolfinx

- `xCFL_number.py`  
  Auxiliary module to compute the CFL number for stability monitoring.

- `useful_functions.py`  
  Utility functions for plotting and post-processing.

## Requirements

- Python >= 3.10
- FEniCSx
- PETSc / petsc4py
- mpi4py
- basix
- gmsh / gmsh-sdk
- tqdm
- numpy
- adios4dolfinx
