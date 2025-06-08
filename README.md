# Wine Quality Prediction with Keras, Hyperopt, and MLflow

This project demonstrates how to build, tune, and track a deep learning model for wine quality prediction using Keras, Hyperopt, and MLflow. The workflow includes hyperparameter optimization, experiment tracking, model registration, and batch prediction.

## Features

- **Data:** Uses the [Wine Quality - White](https://archive.ics.uci.edu/ml/datasets/wine+quality) dataset.
- **Model:** Simple feedforward neural network (ANN) built with Keras.
- **Hyperparameter Tuning:** Uses Hyperopt for learning rate and momentum search.
- **Experiment Tracking:** All runs and metrics are logged to MLflow.
- **Model Serving:** Best model can be loaded and used for batch predictions.

## Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
# or, inside the notebook:
!pip install hyperopt mlflow keras scikit-learn pandas
```

### 2. Run the Notebook

Open `starter.ipynb` in VS Code or Jupyter and run all cells.

### 3. Main Steps

- **Load Data:** Download and split the wine quality dataset.
- **Preprocess:** Split into train, validation, and test sets.
- **Define Model:** Build a Keras ANN with normalization.
- **Tune Hyperparameters:** Use Hyperopt to search for the best learning rate and momentum.
- **Track Experiments:** Log parameters, metrics, and models to MLflow.
- **Compare Runs:** Use the MLflow UI to compare experiment results.
- **Register & Predict:** Load the best model and run predictions on new data.

### 4. MLflow UI

To view your experiments, run:

```bash
mlflow ui
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

## Example Prediction

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

## File Structure

```
starter.ipynb      # Main notebook with all code and workflow
requirements.txt   # (Optional) List of dependencies
README.md          # This file
```

## Notes

- Ensure you only pass feature columns (not the target) to the model for prediction.
- If you encounter permission errors on Windows, try running VS Code as administrator or set `env_manager="local"` in MLflow prediction calls.

## References

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Hyperopt Documentation](http://hyperopt.github.io/hyperopt/)
- [Keras Documentation](https://keras.io/)
- [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)

---
```# Wine Quality Prediction with Keras, Hyperopt, and MLflow

This project demonstrates how to build, tune, and track a deep learning model for wine quality prediction using Keras, Hyperopt, and MLflow. The workflow includes hyperparameter optimization, experiment tracking, model registration, and batch prediction.

## Features

- **Data:** Uses the [Wine Quality - White](https://archive.ics.uci.edu/ml/datasets/wine+quality) dataset.
- **Model:** Simple feedforward neural network (ANN) built with Keras.
- **Hyperparameter Tuning:** Uses Hyperopt for learning rate and momentum search.
- **Experiment Tracking:** All runs and metrics are logged to MLflow.
- **Model Serving:** Best model can be loaded and used for batch predictions.

## Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
# or, inside the notebook:
!pip install hyperopt mlflow keras scikit-learn pandas
```

### 2. Run the Notebook

Open `starter.ipynb` in VS Code or Jupyter and run all cells.

### 3. Main Steps

- **Load Data:** Download and split the wine quality dataset.
- **Preprocess:** Split into train, validation, and test sets.
- **Define Model:** Build a Keras ANN with normalization.
- **Tune Hyperparameters:** Use Hyperopt to search for the best learning rate and momentum.
- **Track Experiments:** Log parameters, metrics, and models to MLflow.
- **Compare Runs:** Use the MLflow UI to compare experiment results.
- **Register & Predict:** Load the best model and run predictions on new data.

### 4. MLflow UI

To view your experiments, run:

```bash
mlflow ui
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

## Example Prediction

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

## File Structure

```
starter.ipynb      # Main notebook with all code and workflow
requirements.txt   # (Optional) List of dependencies
README.md          # This file
```

## Notes

- Ensure you only pass feature columns (not the target) to the model for prediction.
- If you encounter permission errors on Windows, try running VS Code as administrator or set `env_manager="local"` in MLflow prediction calls.

## References

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Hyperopt Documentation](http://hyperopt.github.io/hyperopt/)
- [Keras Documentation](https://keras.io/)
- [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)

---
