# Deep Learning Assignment - I

**Submitted By:** Vijaya Sreyas Badde - AI24BTech11003

## Repository Structure
```
a1
|- Assignment1.ipynb
|
|- data/
|   |- raw/
|   |- noisy/
|
|- report/
|   |- AI2100_Assignment_1.pdf
|   |- dataset_description.pdf
|
|- src/
|   |- mlp.py
|   |- activations.py
|   |- losses.py
|   |- optimizers.py
|   |- adaline.py
|   |- kernels.py
|   |- initializations.py
|
|- plots/
|   |- q1/
|   |- q5/
|   |- q6/
|   |- q7/
|
|- README.md
```
---

## Setup And Configuration

Navigate to your preferred directory in your terminal. Note that you may need to use `python3` in the terminal instead of `python` depending on your device. 

### Set up a virtual environment (ignore if on Linux, mandatory on windows)

Create a virtual environment

```bash
python -m venv venv
```

Activate the virtual environment

```bash
venv\Scripts\Activate
```

### Install Dependencies

Update `pip`
```bash
python -m pip install --upgrade pip
```
This project uses the following `Python` libraries:
```
numpy
matplotlib
scikit-learn
pandas
```
Install them using 
```bash
pip install numpy matplotlib scikit-learn pandas
```
---

## Running the code

### Generate Dataset

Navigate to the assignment's root folder while in the virtual environment, and run the following command:
```bash
python scripts/data_generation.py
```

This will generate a raw dataset in `data/raw/`, and a noisy one in `data/noisy/`. This dataset's name wil be based on the time at which the script is executed (this was added to compare old and new datasets when finetuning the script).

Edit the file paths in `4. Dataset Construction / Loading the Dataset / dataset_raw_path and dataset_noisy_path` in `Assignment1.ipynb` to use the new dataset. You may choose not to, as the previous dataset is still a valid dataset.

### Run Experiments

Open the notebook (`Assignment1.ipynb`) and run all cells. 

This will train models and generate all plots.

---

## Reports
The following reports are included in the `report/` directory:
- **`report/AI2100_Assignment_1.pdf`**: Complete report containing explanations, derivations, and analysis of all experiments.
- **`report/dataset_description.pdf`**: Description of the synthetic dataset generator and process.

---
## Source Code
All model implementations are located in the `src\` directory for modularity.

### Core Models
- [`src/mlp.py`](src/mlp.py)  
  Implementation of a flexible **Multi-Layer Perceptron (MLP)** with backpropagation.

- [`src/adaline.py`](src/adaline.py)  
  Implementation of the **ADALINE** regression model.

### Supporting Modules

- [`src/activations.py`](src/activations.py)  
  Activation functions and their derivatives.

- [`src/losses.py`](src/losses.py)  
  Loss functions used during training.

- [`src/optimizers.py`](src/optimizers.py)  
  Optimizers implemented from scratch:
  - SGD
  - Momentum
  - Nesterov Accelerated Gradient
  - AdaGrad
  - RMSProp
  - Adam
  - Muon

- [`src/kernels.py`](src/kernels.py)  
  Kernel functions used in kernel-based regression methods.

---

## Scripts
The `scripts/` directory contains standalone scripts used for dataset generation and early experiments.

- [`scripts/data_generation.py`](scripts/data_generation.py)  
  Generates the synthetic dataset used throughout the assignment.

- [`scripts/ann_q1.py`](scripts/ann_q1.py)  
  Experiments for Question 1 including XOR learning and cosine function approximation.

---

## Notebook

The main experiments and analysis are contained in [`Assignment1.ipynb`](Assignment1.ipynb). 

The notebook includes:
- Training experiments
- Model comparisons
- Plot Generation
- Kernel method experiments

---

## Plots

Most generated plots are stored in the `plots/` directory.

```
plots/
|- q1/ # ANN experiments
|- q5/ # ADALINE experiments
|- q6/ # MLP Architecture and optimizer studies
|- q7/ # Kernel method experiments
```

Most of these plots were used in the final report for visual analysis.

---

## Implemented Models

The following machine learning models were implemented:

### Neural Networks
- Artificial Neural Network with backpropagation
- Multi-layer perceptron (MLP)

### Linear Models
- ADALINE

### Optimization Algorithms
- SGD
- Momentum
- Adam
- Adagrad
- Nesterov Accelerated GD
- RMSProp
- Muon

### Kernel Methods
- Linear Kernel
- Polynomial Kernel
- RBF Kernel
- Neural Kernel (using learned neural features)

---

## Notes

- All neural networks and optimizers were implemented **from scratch using NumPy**.
- No deep learning frameworks (such as PyTorch or TensorFlow) were used.
- Scikit-learn was used only for the following tasks:
    - t-SNE visualization of learned neural features.
    - Support Vector Regression (SVR) for kernel experiments.
    
    All neural network models, optimizers, activation functions, and training procedures were implemented from scratch usign `NumPy`.
- Python 3.11.9 was used for this assignment.

---