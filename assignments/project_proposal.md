# HWRS640 - Final Project: Proposal Pitch

## Due date: Friday, March 27th at 11:59 PM

## Overview

The final project gives you the opportunity to apply the computational methods from this course to a problem of your own choosing. Over the next 5-6 weeks you will develop a complete data-driven analysis or modeling workflow, culminating in a written report and a short in-class presentation.

This first deliverable is a **~1 page written pitch** that articulates your proposed project. It should be specific enough that the instructor can give you useful feedback before you commit significant time to implementation.

Projects should ideally connect to your own research area, but must not simply reproduce work you have already done or are currently doing — the goal is to apply course methods to a new question, new dataset, or new angle on a familiar problem. New analysis beyond your existing research is also welcome, but may require more careful scoping to be feasible within the time frame.

## Project Requirements

Your project pitch must include all three of the following components:

1. **A dataset** — Real-world Earth science data that you will acquire, process, and analyze. The data should be non-trivial in size or structure (e.g., spatiotemporal fields, multivariate time series, remote sensing imagery). Toy or classroom datasets are not appropriate as the sole data source.
2. **A method** — At least one of the computational methods covered in this course must be central to your analysis. Appropriate methods include (but are not limited to):
  - Dimensionality reduction / EOF / PCA / SVD
  - Dynamic Mode Decomposition (DMD)
  - Deep learning (MLP, CNN, RNN, Transformer, PINN, diffusion model)
  - Physics-informed or hybrid/differentiable modeling
  - Optimization-based approaches
  - Other methods we have not covered but are relevant to data-driven Earth science (discuss with instructor if unsure)
3. **A scientific objective** — A clear question or goal that motivates the analysis. The project should produce a result that is interpretable in an Earth science context, not just a model that runs.

## Proposal Contents (~1 page)

Your pitch should address the following four points. Use these as section headings:

### 1. Motivation and scientific question

   Physically based land surface models such as Noah-MP simulate water and energy fluxes using process-based representations of soil, vegetation, and snow. While these models are physically interpretable, their performance is often limited by uncertain parameterizations and the difficulty of calibrating them across spatially distributed domains.

Recent advances in differentiable modeling and graph neural networks (GNNs) offer a promising framework to bridge physical modeling and data-driven learning. By embedding physical models within a learnable system, it becomes possible to optimize model behavior directly against observations while preserving physical consistency. 

This project explores the following scientific question:
   Can a differentiable implementation of Noah-MP, coupled with a graph neural network routing model informed by topography, improve   streamflow prediction by learning spatially distributed hydrologic behavior across river networks? 

The goal is to develop an end-to-end system that combines physical simulation with data-driven routing to better represent basin-scale hydrologic processes.

### 2. Dataset

The study will focus on the Salt and Verde River basins in Arizona, using high-resolution meteorological, land surface, and hydrologic datasets spanning 1980–2020:

The datasets include:
	•	Meteorological forcing (AORC)
	•	Hourly precipitation, temperature, radiation, and other atmospheric variables
	•	Spatial resolution: ~1 km
	•	Source: NOAA AORC dataset
	•	Static land surface data
	•	Land cover / land mask from MODIS
	•	Vegetation type and properties
	•	Soil type and hydraulic properties from USGS datasets
	•	Topographic information (DEM) for routing and flow direction from SRTM
	•	Hydrologic observations (USGS)
	•	Streamflow (discharge) at selected gauges within the Salt–Verde basin
	•	Used as the target variable for model training and evaluation

#### Key challenges include:
	•	Large spatiotemporal data volume (hourly, multi-decade)
	•	Alignment of forcing, static inputs, and observations
	•	Efficient preprocessing and storage (NetCDF/xarray workflows)
	•	Ensuring that the translation of Noah-MP from Fortran to PyTorch preserves the underlying physical processes while remaining numerically consistent
	•	Maintaining GPU compatibility and computational efficiency in the PyTorch implementation, given the model’s complexity and multiple interacting processes
	•	Managing GPU memory usage to prevent excessive memory consumption during training and simulation
  

### 3. Proposed method

This project applies deep learning and differentiable modeling, focusing on integrating a physically based land surface model with a graph-based routing framework in an end-to-end learning system.

⸻

#### (1) Differentiable Noah-MP (PyTorch)

The first step is to translate Noah-MP from Fortran into PyTorch, enabling automatic differentiation and GPU acceleration. This implementation will preserve the physical structure of the model, including conservation of mass and energy, while allowing gradients to propagate through model states and fluxes.

The model simulates grid-scale hydrologic processes, including:
	•	Soil moisture dynamics across multiple layers
	•	Snow accumulation and melt (snow water equivalent, SWE)
	•	Surface runoff (infiltration excess and saturation excess)
	•	Subsurface drainage and baseflow
	•	Energy balance components affecting evapotranspiration

Inputs:
	•	AORC meteorological forcing (precipitation, temperature, radiation, etc.)
	•	Static land surface properties, including soil type, vegetation type (MODIS), and land cover

Outputs:
	•	Grid-scale runoff components (surface and subsurface)
	•	Hydrologic state variables (soil moisture, SWE)
	•	Energy fluxes (optional for diagnostics)

Special attention will be given to:
	•	Maintaining numerical consistency with the original Fortran model
	•	Ensuring mass conservation across all fluxes
	•	Designing the implementation to be GPU-efficient, avoiding excessive memory usage despite the model’s complexity

⸻

#### (2) Graph Neural Network (GNN) Routing

A graph neural network (GNN) will be used to route runoff through the river network in a physically informed way.
	•	Each grid cell (or aggregated subcatchment) is represented as a node
	•	Graph connectivity is defined using DEM-derived flow directions, ensuring consistency with real topographic flow paths
	•	Edges represent downstream connectivity and flow accumulation pathways

Node features include:
	•	Runoff generated by Noah-MP (time-varying)
	•	Static attributes such as elevation, slope, and upstream contributing area

The GNN will learn how water propagates through the network by:
	•	Aggregating upstream contributions
	•	Learning nonlinear flow transformations and delays
	•	Capturing spatial dependencies that are difficult to represent in traditional routing models

This approach provides a flexible alternative to conventional routing schemes (e.g., kinematic wave or linear reservoirs).

⸻

#### (3) End-to-End Training

The full system will be trained end-to-end, linking physical simulation and learned routing:
	1.	Noah-MP generates grid-scale runoff
	2.	The GNN routes runoff through the river network
	3.	Predicted streamflow at gauge locations is compared with observed USGS discharge

Loss function:
	•	Primary: Mean Squared Error (MSE)
	•	Optional: hydrologically meaningful losses (e.g., NSE-based or log-transformed loss to emphasize low flows)

This setup allows gradients to propagate through:
	•	The GNN routing model
	•	Potentially selected components or parameters of Noah-MP

Training will be conducted over a multi-year period, with careful handling of long sequences to ensure computational feasibility.

### 4. Expected outcomes and evaluation

The expected outcome is a hybrid physics–machine learning framework that improves streamflow prediction while maintaining physically interpretable internal states.

Model performance will be evaluated using standard hydrologic metrics:
	•	Kling–Gupta Efficiency (KGE)
	•	Nash–Sutcliffe Efficiency (NSE)
	•	RMSE and bias

Diagnostic analyses will include:
	•	Observed vs simulated hydrographs at selected gauges
	•	Flow duration curves to assess distributional behavior
	•	Seasonal performance analysis (e.g., winter snowmelt vs summer monsoon)
	•	Spatial comparison of performance across multiple gauges

In addition, the study will evaluate:
	•	Whether GNN-based routing improves streamflow prediction compared to simpler routing approaches
	•	Whether the differentiable Noah-MP framework enables improved calibration of hydrologic processes
	•	Whether internal states (e.g., soil moisture, SWE) remain physically realistic while optimizing streamflow

## Submission Instructions

Submit your proposal as a **PDF or Markdown file** through D2L by the due date. One page is the target; two pages is the hard limit. Figures or diagrams are welcome but not required.

You will receive written feedback within one week. After the proposal, subsequent project milestones are:


| Milestone                                 | Due           |
| ----------------------------------------- | ------------- |
| Project proposal (this assignment)        | Mar 27        |
| Dataset and preprocessing report          | Apr 24        |
| Report on data and model inductive biases | May 1         |
| Final report + presentation               | Week of May 4 |


## Grading

The proposal is graded as part of the final project (20% of course grade). The pitch itself will be evaluated on:


| Criterion                                                       | Weight |
| --------------------------------------------------------------- | ------ |
| Scientific clarity — is the question well-defined?              | 30%    |
| Dataset — is it appropriate and feasible to obtain?             | 25%    |
| Method — is the chosen approach well-motivated?                 | 25%    |
| Feasibility — can this realistically be completed in 5-6 weeks? | 20%    |


## Tips and Scope Guidance

- **Scope down, not up.** A focused project that works is much better than an ambitious project that doesn't. If you are unsure, start simple and add complexity.
- **Data first.** Make sure you can actually access and load your dataset before the proposal is due. Many public datasets have quirks (format, access restrictions, size) that are only discovered when you try to use them.
- **Connection to your research.** You are encouraged to use data from your own research domain, but the analysis must go beyond reproducing existing results. Examples of good reuse: applying DMD to a dataset you normally analyze with other tools; training a neural network on observations you have already collected; using PINNs to emulate a model you normally run.
- **Come to office hours.** If you are uncertain whether your idea is appropriate or feasible, discuss with the instructor before the proposal deadline.

