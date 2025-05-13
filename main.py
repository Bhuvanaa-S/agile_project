import streamlit as st
import pandas as pd
from database import init_db
from preprocessing import preprocess_data
from models import (
    get_ml_model_params,
    configure_dl_model,
    train_and_evaluate_ml_model,
    train_and_evaluate_dl_model
)
from visualization import display_model_comparison

# Initialize Streamlit app
st.title("🧠 Advanced Multi-Model Classifier")

# Initialize database
init_db()

# File upload and data processing
uploaded_file = st.file_uploader("Upload your dataset (CSV only)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📄 *Preview of Data*", df.head())

    # Target selection
    target_column = st.selectbox("🎯 Select target column", df.columns)
    
    # Feature selection
    st.subheader("🔘 Select Features for Modeling")
    all_features = [col for col in df.columns if col != target_column]
    selected_features = st.multiselect(
        "Choose features to include in X", 
        all_features, 
        default=all_features
    )
    
    if not selected_features:
        st.error("Please select at least one feature!")
        st.stop()

    # Preprocess data
    (X_train, X_test, y_train, y_test, 
     X_train_dl, X_test_dl, y_train_cat, y_test_cat, 
     label_encoder, num_classes) = preprocess_data(df, selected_features, target_column)

    # Define models
    ml_models = [
        "Logistic Regression",
        "Random Forest",
        "SVM",
        "KNN",
        "Decision Tree",
        "Naïve Bayes"
    ]

    accuracy_scores = {}

    # Create tabs
    tab1, tab2 = st.tabs(["🧠 Classical ML", "🧬 Deep Learning"])

    with tab1:
        st.subheader("Classical Machine Learning Models")
        selected_ml_models = st.multiselect(
            "Select ML models to train",
            ml_models,
            default=["Random Forest", "SVM"]
        )
        
        for name in selected_ml_models:
            st.subheader(f"🤖 Model: {name}")
            accuracy = train_and_evaluate_ml_model(
                name, X_train, X_test, y_train, y_test, 
                df, label_encoder, target_column
            )
            accuracy_scores[name] = accuracy * 100

        if accuracy_scores:
            display_model_comparison(accuracy_scores)

    with tab2:
        st.subheader("Deep Learning Model")
        accuracy = train_and_evaluate_dl_model(
            X_train_dl, X_test_dl, y_train, y_test, 
            y_train_cat, y_test_cat, df, label_encoder, 
            target_column, X_train_dl.shape[1], num_classes
        )
        accuracy_scores["Deep Neural Network"] = accuracy * 100
        display_model_comparison(accuracy_scores)