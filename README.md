# Optimal Vaccination Strategies on Networks and in Metropolitan Areas

This repository contains code for the project on **Optimal Vaccination Strategies in Metropolitan Areas**, which was the master's project of **Lucas M. Moschen**, supervised by **Maria Soledad Aronna**. 

The research resulted in two papers:
- **Paper 1:** ["Optimal vaccination strategies on networks and in metropolitan areas"](https://doi.org/10.1016/j.idm.2024.06.007) by **María Soledad Aronna** and **Lucas Machado Moschen**.
- **Paper 2:** (To be published)

## Repository Structure

```
📂 root
│── 📜 codes_paper_idm.py                     # Experiments for Paper 1
│── 📜 codes_paper_lcss_ieee                   # Experiments for Paper 2
│── 📜 robot_dance_optimal_control             # Model developed in the dissertation
│── 📜 optimal_control_problem_lcss_ieee_paper.ipynb  # Plots for Paper 2
│── 📜 robot_dance_model_vaccination.ipynb     # Plots for Paper 1 (with vaccination, no optimal control)
│── 📜 robot_dance_model.ipynb                 # Plots for Paper 1 (model without vaccination)
│── 📜 robot_dance_optimal_control.ipynb       # Plots for Paper 1 (with optimal control)
│── 📂 codes/                                  # Source code for models and optimization
│── 📂 data/                                   # Sample datasets (metropolitan areas and commuting patterns)
│── 📂 notebooks/                              # Jupyter notebooks for analysis and visualization
│── 📂 images/                                 # Figures and output from experiments
│── 📜 environment.yml                         # Conda environment file
│── 📜 README.md                               # Project documentation
```

## Overview

This study presents a **mathematical model** for optimal vaccination strategies in interconnected metropolitan areas, considering **commuting patterns**. The model:
- Defines a **vaccination rate for each city** as a control function.
- Integrates **commuting patterns** using a weighted adjacency matrix and a **day/night parameter**.
- Solves an **optimal control problem** minimizing a **cost functional** that balances hospitalizations and vaccine distribution.
- Includes **constraints** on weekly vaccine availability and application capacity.

## Requirements

- **Python 3.8+**
- Required dependencies are listed in [`environment.yml`](./environment.yml).

## Installation & Setup

To set up the environment, use Conda:

```sh
conda env create -f environment.yml
conda activate optimal-vaccine
```

### Updating the Environment
If you modify `environment.yml` (e.g., add/remove dependencies), update the environment with:

```sh
conda env update --file environment.yml --prune
```

## Running Experiments

To reproduce results from the preprint:

1. **Ensure all dependencies are installed.**
2. **Run the Jupyter notebooks** in the `notebooks/` directory to generate the plots and results.

## Citation

If you use this code or data in your research, please cite the preprint:

```
@article{ARONNA20241198,
title = {Optimal vaccination strategies on networks and in metropolitan areas},
journal = {Infectious Disease Modelling},
volume = {9},
number = {4},
pages = {1198-1222},
year = {2024},
issn = {2468-0427},
doi = {https://doi.org/10.1016/j.idm.2024.06.007},
url = {https://www.sciencedirect.com/science/article/pii/S2468042724000897},
author = {M. Soledad Aronna and Lucas Machado Moschen},
keywords = {Optimal control, Epidemiology, Vaccination protocols, Commuting patterns, Metropolitan areas 2000 MSC, 92D30, 49-11},
abstract = {This study presents a mathematical model for optimal vaccination strategies in interconnected metropolitan areas, considering commuting patterns. It is a compartmental model with a vaccination rate for each city, acting as a control function. The commuting patterns are incorporated through a weighted adjacency matrix and a parameter that selects day and night periods. The optimal control problem is formulated to minimize a functional cost that balances the number of hospitalizations and vaccines, including restrictions of a weekly availability cap and an application capacity of vaccines per unit of time. The key findings of this work are bounds for the basic reproduction number, particularly in the case of a metropolitan area, and the study of the optimal control problem. Theoretical analysis and numerical simulations provide insights into disease dynamics and the effectiveness of control measures. The research highlights the importance of prioritizing vaccination in the capital to better control the disease spread, as we depicted in our numerical simulations. This model serves as a tool to improve resource allocation in epidemic control across metropolitan regions.}
}
```
