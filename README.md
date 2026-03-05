# Logistic Regression – Breast Cancer Classification

A practical implementation of **Logistic Regression using Scikit-Learn** to classify tumors as **malignant or benign** using the **Breast Cancer dataset**.

This project demonstrates the **complete machine learning workflow**, including:

- Data loading
- Train-test splitting
- Model training
- Prediction
- Model evaluation
- ROC Curve visualization

---

# Project Overview

Logistic Regression is a **supervised machine learning algorithm used for classification problems**.

Instead of predicting continuous values, it predicts the **probability that an input belongs to a specific class**.

In this project, the model predicts whether a tumor is:

- **Malignant (Cancerous)**
- **Benign (Non-Cancerous)**

The dataset used comes from **Scikit-Learn’s built-in Breast Cancer dataset**.

---

# Dataset Information

Dataset: **Breast Cancer Wisconsin Dataset**

Total samples: **569**

Number of features: **30**

Target Classes:

| Label | Meaning |
|------|--------|
| 0 | Malignant |
| 1 | Benign |

Feature examples include:

- Mean radius
- Mean texture
- Mean perimeter
- Mean area
- Smoothness
- Compactness
- Concavity

---

# Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn

---

# Machine Learning Workflow

The pipeline used in this project:

1. Import required libraries
2. Load dataset
3. Split dataset into training and testing sets
4. Train Logistic Regression model
5. Make predictions
6. Evaluate model performance
7. Visualize ROC Curve

---

# Implementation

## Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
