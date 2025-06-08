# 🍷 Wine Quality Prediction with Keras, Hyperopt, and MLflow

This project demonstrates a complete MLOps workflow for predicting wine quality using a deep learning model. It covers data preparation, model building, hyperparameter tuning, experiment tracking, model registration, and batch prediction—all orchestrated in a reproducible notebook.

---

## 🚀 Project Overview

- **Dataset:** [Wine Quality - White](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **Model:** Feedforward neural network (Keras)
- **Tuning:** Hyperopt for learning rate and momentum
- **Tracking:** MLflow for parameters, metrics, and model artifacts
- **Serving:** Load and predict with the best model

---

## 📂 File Structure

```
starter.ipynb      # Main notebook with all code and workflow
requirements.txt   # (Optional) List of dependencies
README.md          # Project documentation
```

---

## 🛠️ Setup Instructions

### 1. Install Dependencies

You can install all required packages using pip:

```bash
pip install hyperopt mlflow keras scikit-learn pandas
```

Or, run the first cell in the notebook:

```python
!pip install hyperopt
```

### 2. Run the Notebook

Open `starter.ipynb` in VS Code or Jupyter and execute all cells sequentially.

---

## 📊 Workflow Steps

1. **Load Data:**  
   Download and preview the wine quality dataset.

2. **Preprocess:**  
   - Split into train, validation, and test sets.
   - Separate features and target (`quality`).

3. **Model Definition:**  
   - Build a Keras ANN with normalization and dense layers.

4. **Hyperparameter Tuning:**  
   - Use Hyperopt to search for the best learning rate and momentum.
   - Each trial is tracked as an MLflow run.

5. **Experiment Tracking:**  
   - Log parameters, metrics, and models to MLflow.
   - Compare runs and select the best model.

6. **Model Registration & Prediction:**  
   - Load the best model from MLflow.
   - Predict wine quality on new data (excluding the `quality` column).

---

## 📈 Using MLflow UI

To visualize and compare experiment runs, launch the MLflow UI:

```bash
mlflow ui
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 🧪 Example: Batch Prediction

```python
import mlflow
logged_model = 'runs:/<run_id>/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame (excluding the "quality" column)
import pandas as pd
feature_data = data.drop("quality", axis=1)
predictions = loaded_model.predict(feature_data)
```

---

## ⚠️ Troubleshooting

- **Windows Permission Errors:**  
  If you see permission errors (e.g., symlink or file access issues), try:
  - Running VS Code or Jupyter as administrator.
  - Setting `env_manager="local"` in MLflow prediction calls.

- **Prediction Shape Errors:**  
  Always pass only the feature columns (not the target) to the model for prediction.

---

## 📚 References

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Hyperopt Documentation](http://hyperopt.github.io/hyperopt/)
- [Keras Documentation](https://keras.io/)
- [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)


---

**Happy Experimenting!**
