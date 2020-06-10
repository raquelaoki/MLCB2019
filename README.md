# Project - Spring 2019

## Project Part 1 

Submitted to 14th Machine Learning in Computational Biology (MLCB) meeting, co-located with NeurIPS in Vancouver (2019)

Aoki, R., and Martin, E.. (Poster) Bayesian Predictive Model combined with Matrix Factorization for Causal Inference Analysis. 14th Machine Learning in Computational Biology (MLCB) meeting, co-located with NeurIPS, 2019.

## Summary 

A common limitation in causal inference is the influence of confounders, variables that might influence features and response variables, but are not present in the dataset or cannot be directly observed. Aiming to find alternatives, in this project, we present a variation of the Deconfounder algorithm, through a probabilistic graphical model that combines a factor analysis and a predictive model. The idea is to capture the effect of confounders with factor analysis, and use the predictive model to evaluate the quality of the latent variables; later, we make the causal inference analysis using Outcome Models. This approach is currently a work in progress.

## Code description 

This project contains all the codes used for the experiments and implementation. 
The data download and pre processing is in R language. Feature extraction for the DA are in Python, while BART and GFCI are in R. 

## Data Exploration 

Source: https://www.ncbi.nlm.nih.gov/clinvar/
