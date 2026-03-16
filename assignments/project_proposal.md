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

   Rainfall–runoff models attempt to represent how precipitation and other climate drivers generate streamflow in river basins. Recent machine learning approaches can achieve high predictive skill but often lack physical interpretability, while traditional hydrologic models rely on parameters that are difficult to estimate for ungauged basins. Hybrid approaches that combine physical models with data-driven parameter estimation offer a promising path forward.

   This project explores whether graph neural networks (GNNs) can regionalize physically interpretable hydrologic model parameters across many catchments. Specifically, the scientific question is:
   Can a graph neural network learn relationships between catchment characteristics and hydrologic model parameters such that a physically based rainfall–runoff model can be applied to previously unseen basins?
   Answering this question could improve prediction in ungauged basins while maintaining interpretable hydrologic processes.

### 2. Dataset

Describe the data you plan to use:

- What it is (variable(s), domain, temporal/spatial resolution)
- Where it comes from (source, how you will access it)
- Any known challenges (missing data, large file sizes, preprocessing needed)
   The analysis will use the CAMELS-US (Catchment Attributes and Meteorology for Large-sample Studies) dataset, which contains hydrologic and meteorological data for hundreds of catchments across the continental United States.
   The dataset includes:
  •    Daily meteorological forcing: precipitation, temperature, and potential evapotranspiration
  •    Observed streamflow: daily discharge records from USGS gauges
  •    Catchment attributes: topography, climate statistics, soil properties, vegetation cover, and geology
  •    Spatial coverage: ~500 catchments across diverse hydroclimatic regions in the U.S.
  •    Temporal resolution: daily data spanning multiple decades
   The dataset is publicly available and can be accessed through the CAMELS data repository. Preprocessing will involve aligning meteorological forcing with streamflow records and normalizing catchment attributes.
   One challenge is the large number of basins and long time series, which can make training computationally intensive. To address this, catchments will be split into training, validation, and test sets (e.g., 300/100/100 basins).

### 3. Proposed method

Describe the computational approach you plan to apply:

- Which method(s) from the course will you use, and why are they appropriate for this problem?
- What will the inputs and outputs of your model or analysis be?
- If you are using deep learning, describe the architecture you have in mind (even if preliminary).



   This project will apply deep learning and hybrid/differentiable modeling, two of the computational approaches discussed in this course.
   The proposed workflow consists of two components:
   Graph Neural Network (GNN)
   Each catchment will be represented as a node in a graph, where node features consist of static catchment attributes (e.g., area, slope, soil properties, climate statistics). Edges between nodes will represent hydrologic similarity between catchments based on attribute distance.
   The GNN will learn a mapping from catchment attributes to hydrologic model parameters


   Thus, the network predicts a set of parameters for each basin.
   Hydrologic Model (MCP)
   The predicted parameters will be used within a Mass-Conserving Perceptron (MCP) rainfall–runoff model. Two variants of the MCP model will be evaluated:
   •    M5 with ponding
   •    M5 with ponding and vertical drainage
   For each basin, the MCP model will simulate streamflow using meteorological forcing. The difference between simulated and observed streamflow will be used to compute the training loss.

I  nductive Training Strategy

   To evaluate generalization, an inductive learning setup will be used. Catchments will be divided into training, validation, and testing groups. The GNN will be trained only using the training basins, while validation and test basins will remain unseen during training. This allows evaluation of the model’s ability to estimate parameters for new catchments.

### 4. Expected outcomes and evaluation

   What do you expect to find? How will you know if your method worked? Describe at least one concrete evaluation metric or diagnostic plot you plan to produce.

   The primary outcome will be a trained model capable of predicting hydrologic parameters for unseen basins and generating streamflow simulations using the MCP model.

   Model performance will be evaluated using standard hydrologic metrics, including:
      •	Kling–Gupta Efficiency (KGE)
      •	Nash–Sutcliffe Efficiency (NSE)

   These metrics will be computed for both validation and test basins.

   In addition to summary metrics, several diagnostic plots will be produced:
      •	Observed vs simulated hydrographs for selected basins
      •	Flow duration curves comparing observed and simulated streamflow
      •	Spatial maps showing model performance across basins

   Comparisons between the two MCP model structures (with and without drainage) will help assess how additional physical processes influence predictive performance.

   Overall, the project will demonstrate whether graph-based parameter regionalization combined with a physics-based hydrologic model can improve rainfall–runoff prediction in previously unseen catchments.

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

