# Diabetes Detection using Logistic Regression (from scratch)

This project demonstrates how **logistic regression** works by implementing it **from scratch in Python**, using only `pandas` and `math`.  
The model is trained on the **Pima Indians Diabetes Dataset** (available on Kaggle), with no use of machine learning libraries such as `scikit-learn`.

## 📊 Dataset
- **Name**: Pima Indians Diabetes Database
- **Source**: [Kaggle Link](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Samples**: 768 individuals
- **Features**: 8 medical features + 1 target (`Outcome`)

## 🛠 Features of the project
- Manual implementation of logistic regression
- Gradient descent for weight optimization
- Elbow method for early stopping
- Final prediction example on a new individual

## 📁 Files
- `logistic_regression_diabetes.py` – contains the full implementation
- `diabetes.csv` – dataset used for training
- `README.md` – this file

## 🧠 Why from scratch?
Building machine learning models from scratch helps understand how things work under the hood — including loss functions, weight updates, and model convergence.

## ✅ Requirements
- Python 3.x
- pandas
- matplotlib (for plotting loss curve)

Install required libraries:
```bash
pip install pandas matplotlib
