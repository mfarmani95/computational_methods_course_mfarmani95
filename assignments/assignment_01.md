# HWRS640 - Assignment 1: Computer architecture and parallel computing

## Due date: Friday, February 6th at 11:59 PM

## Problem 1: Supercomputer architecture (25 points)

The top 500 supercomputers in the world are ranked based on their performance on the LINPACK benchmark, which measures a system's floating-point computing power. Visit the [Top500 website](https://www.top500.org/lists/top500/) and select one of the top 50 supercomputers. Write a short report (~0.5-1 page) that includes the following information:

1. The name and location of the supercomputer.
2. The architecture of the supercomputer (e.g., CPU type, number of cores, memory, interconnect).
3. The peak performance of the supercomputer in FLOPS (floating-point operations per second).
4. A brief discussion of the applications or research areas that benefit from this supercomputer's capabilities.


## Supercomputer: El Capitan

### 1. Name and Location
The supercomputer selected is **El Capitan**, currently ranked #1 on the TOP500 list. It is located at the **Lawrence Livermore National Laboratory (LLNL)** in **Livermore, California, United States**, and is operated by the U.S. Department of Energy’s National Nuclear Security Administration (NNSA).

---

### Architecture
El Capitan is built on the **HPE Cray EX255a** architecture and integrates **AMD 4th Generation EPYC CPUs** with **AMD Instinct MI300A accelerators**. The system consists of more than **11 million combined CPU and GPU cores**, designed to support extreme parallelism. Compute nodes use a shared-memory model between CPU and GPU components and are connected via the **HPE Slingshot interconnect**, which provides high bandwidth and low latency communication across the system.

The system operates using a Linux-based operating system optimized for high-performance computing environments and includes petabytes of distributed system memory to support large-scale simulations.

---

### Peak Performance
On the LINPACK benchmark used by the TOP500 ranking, El Capitan achieves a sustained peak performance of approximately **1.74 exaFLOPS** (1.74 × 10¹⁸ floating-point operations per second). Its theoretical peak performance exceeds **2.8 exaFLOPS**, placing it firmly within the exascale computing class.

---

### Applications and Research Areas
El Capitan is primarily used to support **national security and stockpile stewardship**, enabling high-fidelity simulations to ensure the safety and reliability of the U.S. nuclear stockpile without physical testing. Beyond national defense, the system supports a wide range of scientific research areas, including **climate and Earth system modeling**, **materials science**, **biological simulations**, and **large-scale artificial intelligence–assisted modeling**. Its computational capabilities allow researchers to perform simulations and analyses that were previously computationally infeasible.

---



## Problem 2: Moore's Law and Linear Regression (25 points)
Moore's Law states that the number of transistors on a microchip doubles approximately every two years, leading to an exponential increase in computing power. Using the provided historical data, given in `computational_methods_course/data/moores.csv`, perform the following tasks:

1. Load the data into a pandas DataFrame.
2. Use linear regression to model the relationship between the year and the number of transistors.
3. Plot the original data points and the fitted regression line.
4. Compute the doubling time of transistors based on your regression model, and compare it to the commonly cited value of two years.
5 (for fun). Compute the same regression for the first 10 years of the data and the last 10 years of the data. Has the doubling time changed over the history of computing?

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_csv('computational_methods_course/data/moores.csv')
print(df.head(5))
```


## Problem 3: Row vs column order data access (25 points)
In this problem, you will explore the performance differences between row-major and column-major data access patterns using NumPy arrays. Perform the following tasks:

1. Create a large 2D NumPy array (e.g., 10,000 x 10,000) filled with random numbers from a distribution of your choosing.
2. Implement two functions to compute the sum of all elements in the array using python loops:
   - One function that accesses the array in row-major order.
   - Another function that accesses the array in column-major order.
3. Measure and compare the execution time of both functions using the `time` module or `timeit` library. Make sure to repeat the measurements multiple times (at least 30) to get an average execution time.
4. Compare the performance results to using built in NumPy functions for summing the array. Explain the differences in performance you observe, using concepts such as cache locality and memory access patterns.

## Problem 4: Scaling and parallel computing (25 points)

In this problem, you will use Dask arrays to compute the element-wise standard score (z-score normalization) of a large random array and measure the scaling behavior across 1-4 CPU cores. The z-score is computed as: z = (x - μ) / σ, where μ is the mean and σ is the standard deviation.

Perform the following tasks:

1. Create a function that generates a large Dask array filled with random numbers and computes the z-score normalized array.
2. **Strong scaling**: Fix the array size (e.g., 20,000 x 20,000) and measure execution time using 1, 2, 3, and 4 cores. Calculate the speedup S(p) = T(1)/T(p) and efficiency E(p) = S(p)/p. Plot execution time vs number of cores.
3. **Weak scaling**: Scale the array size proportionally with the number of cores (maintaining constant work per core). Measure execution time for 1-4 cores and plot the results.
4. Discuss your results: Does your implementation achieve good scaling? What factors limit the speedup?

**Hint**: Configure the number of workers using `dask.config.set(num_workers=n)` and use `.compute()` to trigger computation.
