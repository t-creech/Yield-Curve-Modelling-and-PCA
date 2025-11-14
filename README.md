# Yield Curve Modelling and PCA  

This project aims to model the yield curve and analyze its dynamics using principal component analysis (PCA).  The yield curve describes how interest rates vary across different maturities; understanding its shape is important for pricing fixed‑income instruments and managing interest‑rate risk.  By decomposing the curve into a few orthogonal factors, we can capture most of its variation with simple components.  

## Features  

- **Data ingestion & cleaning** – scripts to download and clean yield data.  
- **PCA implementation** – compute principal components of the yield curve to identify level, slope and curvature factors.  
- **Visualization & analysis** – notebooks or scripts to visualize the yield curve over time and interpret the PCA factors.  
- **Makefile & environment management** – reproducible workflow with a `Makefile` and `environment.yml` for creating the development environment.  

## Repository structure  

- `src/` – source code for data preparation, PCA computation and analysis.  
- `notebooks/` – interactive notebooks exploring the yield curve data and PCA results.  
- `Makefile` – defines tasks for setting up the environment, running analysis, etc.  
- `environment.yml` – list Python dependencies.  

## Getting started  

1. Clone the repository:  
   ```bash  
   git clone https://github.com/t-creech/Yield-Curve-Modelling-and-PCA.git  
   cd Yield-Curve-Modelling-and-PCA  
   ```  

2. Create a conda environment:  
   ```bash  
   conda env create -f environment.yml  
   conda activate yield-curve-pca  
   ```  

3. Run the analysis (example):  
   ```bash  
   make all  
   ```  
   or run the scripts in `src/` manually.  

## Goals  

- Understand the term structure of interest rates through factor decomposition.  
- Demonstrate how PCA can be applied to financial time‑series.  
- Provide a reproducible example of a small research project in Python.  

Feel free to explore the code and adapt it to your own fixed‑income datasets.
