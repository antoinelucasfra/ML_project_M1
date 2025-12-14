# ML Project M1 - Heart Disease Classification

Machine learning project from M1 Data Science, Agrocampus Ouest (March 2019).

**Authors:** Antoine Lucas, Gabriel Besombes

## Overview

Binary classification project to predict heart disease using patient health data. We compare several ML algorithms:

- **K-Nearest Neighbors (KNN)** - with hyperparameter tuning for k
- **Logistic Regression** - multinomial with LBFGS solver
- **Naive Bayes** - Gaussian implementation
- **Support Vector Machine (SVM)** - linear kernel

## Languages

This project is available in two languages:

| Language | Notebook | Python Script |
|----------|----------|---------------|
| 🇬🇧 English | `heart_disease_classification.ipynb` | `heart_disease_classification.py` |
| 🇫🇷 French | *(original)* | `classification_maladie_cardiaque.py` |

## Data

Dataset: [Heart Disease UCI](https://www.kaggle.com/ronitf/heart-disease-uci)

The `heart.csv` file contains patient features and a binary target indicating heart disease presence.

## Setup

### Option 1: Using pip

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using uv (recommended)

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Option 3: Using pyproject.toml

```bash
pip install -e .
```

## Usage

### Run the Jupyter Notebook

```bash
jupyter notebook machine_learning_M1_project.ipynb
```

Or open `machine_learning_M1_project.ipynb` directly in VS Code / Positron.

### Run the Python script

```bash
# English version
python heart_disease_classification.py

# French version
python classification_maladie_cardiaque.py
```

## Project Structure

```
ML_project_M1/
├── heart.csv                              # Dataset
├── heart_disease_classification.ipynb     # Notebook (English)
├── heart_disease_classification.py        # Python script (English)
├── classification_maladie_cardiaque.py    # Python script (French)
├── gabriel_antoine_projet.html            # Rendered HTML report (French)
├── requirements.txt                       # Python dependencies
├── pyproject.toml                         # Project metadata
├── .gitignore                             # Git ignore rules
└── README.md
```

## Results

Best performing models achieved ~90% accuracy. Logistic Regression and SVM showed more consistent performance than KNN on this dataset. See the notebook for detailed confusion matrices and analysis.
