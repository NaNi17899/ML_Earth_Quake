# ML Earthquake Prediction

A Google Colab-based machine learning project for predicting earthquake alert levels from seismic features. The notebook/script explores several classifiers, trains a Random Forest model, saves the trained artifacts to Google Drive, and launches a Gradio app for interactive predictions.

## Overview

This project uses a balanced earthquake dataset to classify the alert level of an event into one of four labels:

- Green
- Yellow
- Orange
- Red

The workflow includes:

- Loading the dataset from Google Drive
- Inspecting missing values and class balance
- Encoding the target label
- Ranking features with chi-square statistics
- Training and comparing multiple ML models
- Selecting and saving a Random Forest model
- Building a Gradio interface for live predictions

## Project Features

- Multi-model evaluation with KNN, SVM, Bagging, Random Forest, AdaBoost, Gradient Boosting, XGBoost, and Stacking
- Feature scaling with `StandardScaler`
- Label encoding with `LabelEncoder`
- Confusion matrix and classification report output
- Feature importance visualization
- Model export with `joblib`
- Simple web interface using Gradio

## Dataset

The script expects a CSV file named:

- `earthquake_alert_balanced_dataset.csv`

Expected location in Google Drive:

- `/content/drive/MyDrive/earthquake_alert_balanced_dataset.csv`

Target column:

- `alert`

Feature columns used by the final prediction function:

- `magnitude`
- `depth`
- `cdi`
- `mmi`
- `sig`

## Requirements

The code was written for Google Colab and uses the following Python packages:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- joblib
- gradio

## Setup

### In Google Colab

1. Upload the dataset to your Google Drive.
2. Make sure the CSV is available at:
   - `/content/drive/MyDrive/earthquake_alert_balanced_dataset.csv`
3. Run the notebook or Python cells in Colab.
4. Allow Drive access when prompted.

### If running locally

The current script uses Colab-specific paths and `drive.mount()`. To run it locally, update the file paths and remove the Google Drive mount code.

## How It Works

### 1. Load and inspect data

The dataset is loaded from Google Drive and then summarized with:

- `df.head()`
- `df.info()`
- missing value counts
- class distribution checks

### 2. Encode the target

The `alert` column is encoded into numeric labels with `LabelEncoder`.

### 3. Feature ranking

Chi-square scores are calculated to estimate feature importance.

### 4. Train/test split

The dataset is split into training and test sets using an 80/20 split with stratification.

### 5. Scale features

`StandardScaler` is used before training the models.

### 6. Model comparison

Several models are trained and evaluated, and their accuracies are compared.

### 7. Final training

A Random Forest classifier is trained as the final model and evaluated with:

- accuracy score
- classification report
- confusion matrix

### 8. Save artifacts

The trained model, scaler, and label encoder are saved to Google Drive as:

- `earthquake_alert_model_fixed.pkl`
- `earthquake_alert_scaler_fixed.pkl`
- `earthquake_alert_encoder.pkl`

### 9. Gradio app

A Gradio interface is launched to predict the earthquake alert level from:

- magnitude
- depth
- CDI
- MMI
- significance (`sig`)

## Running the Predictor

Once the model files are saved to Google Drive, the Gradio app loads them and returns a formatted prediction such as:

- Green (Minor)
- Yellow (Moderate)
- Orange (Strong)
- Red (Severe)

## Notes

- The script contains repeated training cells, which look like iterative experimentation in Colab.
- The final working pipeline is the Random Forest model saved with the `_fixed` filenames.
- `iface.launch(share=True)` creates a public Gradio link when running in Colab.

## Output Files

The code saves these artifacts:

- model file: `earthquake_alert_model_fixed.pkl`
- scaler file: `earthquake_alert_scaler_fixed.pkl`
- label encoder: `earthquake_alert_encoder.pkl`

## Example Usage

After launching the Gradio app, enter values for the five input features and the app will return the predicted alert level.

## License

Add a license here if you want to publish or share the project.
