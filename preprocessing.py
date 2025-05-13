import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from scipy.sparse import issparse
from tensorflow.keras.utils import to_categorical
import streamlit as st

def is_text_column(series):
    if series.dtype == object:
        sample = series.dropna().head(10)
        return any(isinstance(x, str) and any(c.isalpha() for c in str(x)) for x in sample)
    return False

def is_categorical_text(series):
    unique_count = series.dropna().nunique()
    return 1 < unique_count < 20

def is_categorical_numeric(series):
    return pd.api.types.is_numeric_dtype(series) and 1 < series.nunique() < 20

def preprocess_data(df, selected_features, target_column):
    X = df[selected_features]
    y = df[target_column]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)

    text_cols = [col for col in X.columns if is_text_column(X[col])]
    cat_num_cols = [col for col in X.columns if is_categorical_numeric(X[col]) and col not in text_cols]
    pure_num_cols = [col for col in X.select_dtypes(include=np.number).columns 
                    if col not in cat_num_cols and col not in text_cols]

    final_features = []
    feature_info = []

    for col in text_cols:
        col_data = X[col].fillna('')
        if is_categorical_text(col_data):
            le = LabelEncoder()
            X[col] = le.fit_transform(col_data)
            final_features.append(X[[col]])
            feature_info.append(f"Text (categorical - label encoded): {col}")
        else:
            tfidf = TfidfVectorizer(max_features=100)
            tfidf_result = tfidf.fit_transform(col_data)
            tfidf_df = pd.DataFrame(tfidf_result.toarray(), 
                                  columns=[f"{col}_{word}" for word in tfidf.get_feature_names_out()])
            final_features.append(tfidf_df)
            feature_info.append(f"Text (TF-IDF vectorized): {col} ({tfidf_df.shape[1]} features)")

    if cat_num_cols:
        num_cat_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('scaler', MinMaxScaler())
        ])
        num_cat_data = num_cat_transformer.fit_transform(X[cat_num_cols])
        num_cat_df = pd.DataFrame(num_cat_data, columns=cat_num_cols)
        final_features.append(num_cat_df)
        feature_info.append(f"Numeric categorical (min-max scaled): {', '.join(cat_num_cols)}")

    if pure_num_cols:
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        numeric_data = numeric_transformer.fit_transform(X[pure_num_cols])
        numeric_df = pd.DataFrame(numeric_data, columns=pure_num_cols)
        final_features.append(numeric_df)
        feature_info.append(f"Continuous numeric (standard scaled): {', '.join(pure_num_cols)}")

    X_all_transformed_df = pd.concat(final_features, axis=1)
    
    st.subheader("🔍 Feature Processing Summary")
    for info in feature_info:
        st.write(f"✅ {info}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_all_transformed_df, y_encoded, test_size=0.2, random_state=42)
    
    X_train_dl = X_train.values if not issparse(X_train) else X_train.toarray()
    X_test_dl = X_test.values if not issparse(X_test) else X_test.toarray()
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    return (X_train, X_test, y_train, y_test, 
            X_train_dl, X_test_dl, y_train_cat, y_test_cat, 
            label_encoder, num_classes)