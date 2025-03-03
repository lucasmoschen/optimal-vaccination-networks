# Optimal Vaccination Strategies on Networks and in Metropolitan Areas

This repository contains the code and data accompanying the paper "[Optimal vaccination strategies on networks and in metropolitan areas](https://doi.org/10.1016/j.idm.2024.06.007)" by María Soledad Aronna and Lucas Machado Moschen.

## Overview

The study presents a mathematical model for optimal vaccination strategies in interconnected metropolitan areas, considering commuting patterns. 
The model incorporates a vaccination rate for each city, acting as a control function, and integrates commuting patterns through a weighted adjacency matrix and a parameter that distinguishes day and night periods. 
The optimal control problem aims to minimize a cost functional that balances the number of hospitalizations and vaccines, including constraints on weekly vaccine availability and application capacity.

## Contents

- **`codes/`**: Source code for the mathematical models and optimization algorithms.
- **`data/`**: Sample datasets representing metropolitan areas and commuting patterns.
- **`notebooks/`**: Jupyter notebooks demonstrating the analysis and visualization of results.
- **`images/`**: Output from experiments.

## Requirements

- Python 3.8 or higher
- Required Python packages are listed in `requirements.yml`.

## Reproducing Preprint Results

To reproduce the experiments and figures from the preprint:

1. Install all the necessary libraries.
2. Execute the notebooks in the `notebooks/` directory.

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
