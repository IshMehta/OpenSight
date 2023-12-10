# OpenSight: Leveraging Explainable AI for Targeted Congestion Reduction and Pollution Mitigation

## Air Quality Prediction

This repository contains Python scripts for predicting air quality based on environmental and vehicle-related indicators. The project involves data preprocessing, including merging environmental and vehicle data, and building a neural network model for prediction. Additionally, the repository integrates the explainable AI method, SHAP (SHapley Additive exPlanations), to highlight influential features and provide insights into the model's predictions.

## Prerequisites

Before running the code, ensure you have the following installed:

- Python (3.x)
- pip (Python package installer)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/DragonCoderz/SDG.git
    ```

2. Navigate to the project directory:

    ```bash
    cd SDG
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Running the Code

### Step 1: Data Preprocessing (createpivot.py)

Run the following command to preprocess the data and generate the pivot file:

```bash
python createpivot.py
```

This script loads the 'Air_Quality.csv' dataset, filters relevant indicators, and creates a pivot file ('pivot.csv') for use in the prediction model.

### Step 2: Neural Network Prediction (model2.py)

Run the following command to train the neural network model and make predictions:

```bash
python model2.py
```

This script reads the pivot file ('pivot.csv'), preprocesses the data, builds a neural network model, trains the model, and evaluates its performance. It also integrates SHAP for explainability.

Note: Ensure that you have completed Step 1 before running Step 2, as 'model2.py' relies on the pivot file generated in the preprocessing step.

## Understanding the Code

- **createpivot.py**: This script performs data preprocessing, filtering relevant indicators, and creating a pivot file. It prepares the data for the neural network model.

- **model2.py**: This script reads the pivot file, preprocesses the data, builds a neural network model using TensorFlow and Keras, trains the model, and evaluates its performance. It integrates SHAP for explainability.

## SHAP Integration

This repository uses the SHAP (SHapley Additive exPlanations) method for explainable AI. SHAP values provide insights into the model's predictions by highlighting influential features and their impact on individual predictions.

## Bigger Picture

This repository only represents one possible instance of the backend of our desired product. We have also created a figma mockup to serve as a sample of what our front end could look like. You can find it [here](https://www.figma.com/proto/shO3D6uucWPRrNmpUELZlc/Untitled?type=design&node-id=18-121&t=cupqisaOcji3dT0U-1&scaling=scale-down&page-id=0%3A1&starting-point-node-id=1%3A4&mode=design)
