# Advanced Multi-Model Classifier

A Streamlit-based web application for training, evaluating, and comparing classical machine learning and deep learning models on user-uploaded datasets. This tool provides an interactive interface for data preprocessing, model selection, hyperparameter tuning, and result visualization, with caching to optimize repeated runs.

## Features

- **Data Upload:** Upload CSV datasets for analysis.
- **Flexible Preprocessing:**
  - Handles numerical, categorical, and text features.
  - Applies TF-IDF for text.
  - Applies label encoding for categorical data.
  - Applies scaling for numerical data.
- **Model Support:**
  - **Classical ML:** Logistic Regression, Random Forest, SVM, KNN, Decision Tree, Naïve Bayes.
  - **Deep Learning:** Configurable neural networks with customizable layers, units, activation functions, and dropout.
- **Hyperparameter Tuning:** Adjust model parameters through an intuitive UI.
- **Evaluation Metrics:** Displays accuracy, precision, recall, F1-score, per-class metrics, and confusion matrices.
- **Result Caching:** Stores model results in a SQLite database to avoid redundant computations.
- **Visualization:** Includes confusion matrices, training history plots (for deep learning), and model comparison charts.
- **Interactive Interface:** Streamlit tabs for classical ML and deep learning workflows.

## Requirements

- Python 3.8 or higher
- SQLite3 (included with Python)
- Required Python packages (listed in `requirements.txt`):
  - `streamlit`
  - `pandas`
  - `numpy`
  - `seaborn`
  - `matplotlib`
  - `scikit-learn`
  - `tensorflow`
  - `protobuf==3.20.*`

## Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/Bhuvanaa-S/agile_project.git
    cd agile_project
    ```

2.  **Set Up a Virtual Environment (recommended):**

    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit App:**

    ```bash
    streamlit run app.py
    ```

2.  **Interact with the App:**
    - Open the provided URL (usually `http://localhost:8501`) in your browser.
    - Upload a CSV dataset.
    - Select the target column and features for modeling.
    - Choose classical ML models and/or configure a deep learning model.
    - Adjust hyperparameters as needed.
    - View model performance metrics, confusion matrices, and comparison charts.

## Data Requirements

- **Format:** CSV file with a header row.
- **Columns:** At least one target column (for classification).
- **Features:** Can be numerical, categorical, or text-based.
- **Target:** Should contain discrete classes (binary or multi-class).
- **Text Columns:**
  - Handled via TF-IDF for non-categorical text.
  - Handled via label encoding for categorical text with fewer than 20 unique values.
- **Missing Values:** Automatically handled via imputation:
  - Median for numerical features.
  - Most frequent value for categorical features.
