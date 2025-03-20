# Feature Selection with Nearest Neighbor

This project implements a simple feature selection algorithm using a nearest neighbor classifier. It provides two search strategies for selecting the best subset of features from a dataset with continuous attributes and two classes:

- **Forward Selection:** Starts with an empty feature set and iteratively adds features that improve classification accuracy.
- **Backward Elimination:** Starts with the full feature set and iteratively removes features that yield the best accuracy improvement.

The nearest neighbor classifier is used with a leave-one-out strategy to evaluate the performance of each feature subset.


## File Structure

- **instance.h / instance.cpp**  
  Defines the `Instance` class that stores the class label and the continuous features for each data record.

- **problem.h / problem.cpp**  
  Contains the `Problem` class which wraps the dataset and implements the nearest neighbor classification method. It provides functions such as `dataset_size()`, `Nearest_N()`, and helper functions to calculate Euclidean distances and majority class accuracy.

- **main.cpp**  
  The main driver file that:
  - Parses the input data file.
  - Creates a `Problem` object using the dataset.
  - Prompts the user to choose a feature selection algorithm (Forward Selection, Backward Elimination, or a special algorithm).
  - Executes the selected algorithm and displays the trace of feature subset evaluations.

- **trace.txt**  
  A sample output file that shows the detailed progress of the feature selection process (the trace). Two example formats are provided below – one for forward selection and one for backward elimination.


## Project Overview

### Nearest Neighbor Classifier
- **Purpose:**  
  Evaluate the accuracy of a feature subset by classifying instances using a nearest neighbor classifier with leave-one-out cross-validation.
- **Key Points:**  
  - If no features are selected, the classifier uses the majority class accuracy.
  - Euclidean distance is calculated using only the selected features.

### Feature Selection Algorithms
- **Forward Selection:**  
  - Starts with an empty set.
  - In each iteration, adds one feature (not already selected) that leads to the highest increase in accuracy.
  - Prints the feature subset and its accuracy at every step.
  - Warns when accuracy decreases.
- **Backward Elimination:**  
  - Starts with the full feature set.
  - In each iteration, removes one feature that, when eliminated, gives the highest accuracy.
  - Similarly prints the trace of feature subsets and warns when accuracy decreases.


## Requirements

- **C++ Compiler:**  
  A modern C++ compiler that supports C++11 or later (e.g., g++, clang++).
- **Standard Libraries:**  
  The project makes use of standard libraries (`iostream`, `vector`, `queue`, etc.) and does not require additional third-party libraries.


## Building the Project

1. **Compile the Code:**  
   Use your preferred C++ compiler. For example, with g++:
   ```bash
   g++ -std=c++11 -o feature_selection main.cpp instance.cpp problem.cpp
   ```
2. **Prepare Your Dataset:**  
   Create a data file (e.g., `data.txt`) formatted as follows:  
   - The first column is the class label (1 or 2).  
   - The remaining columns are continuous feature values.
   
   Example (one instance per line):
   ```
   2.0000000e+00 1.2340000e+10 2.3440000e+00
   1.0000000e+00 6.0668000e+00 5.0770000e+00
   ```


## Running the Project

Run the compiled executable and pass your data file as a command-line argument:
```bash
./feature_selection data.txt

Upon running, you will see a prompt similar to:

Welcome to Anisha's Feature Selection Algorithm
Type the number of the selection algorithm you want
     1) Forward Selection
     2) Backward Elimination
     3) Anisha's Special Algorithm

Enter your choice (1, 2, or 3) to start the corresponding feature selection process.
