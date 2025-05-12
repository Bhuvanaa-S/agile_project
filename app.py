import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import hashlib
import sqlite3
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from scipy.sparse import issparse

# Deep learning imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Initialize Streamlit app
st.title("🧠 Advanced Multi-Model Classifier")

# Database functions (unchanged)
def get_df_hash(df):
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

def init_db():
    conn = sqlite3.connect("ml_results.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS models_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            df_hash TEXT,
            model_type TEXT,
            model TEXT,
            accuracy REAL,
            precision REAL,
            recall REAL,
            f1 REAL,
            conf_matrix TEXT,
            per_class_precision TEXT,
            target TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_metrics_to_db(df_original, model_type, model_name, metrics_dict, conf_matrix, per_class_precision, target):
    df_hash = get_df_hash(df_original)
    conn = sqlite3.connect("ml_results.db")
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO models_metrics (
            df_hash, model_type, model, accuracy, precision, recall, f1, conf_matrix, 
            per_class_precision, target
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        df_hash,
        model_type,
        model_name,
        metrics_dict['accuracy'],
        metrics_dict['precision'],
        metrics_dict['recall'],
        metrics_dict['f1_score'],
        json.dumps(conf_matrix.tolist()),
        json.dumps(per_class_precision.tolist()),
        target
    ))
    conn.commit()
    conn.close()

def load_metrics_if_exist(df_original, model_type, model_name, target):
    df_hash = get_df_hash(df_original)
    conn = sqlite3.connect("ml_results.db")
    cursor = conn.cursor()
    cursor.execute('''
        SELECT accuracy, precision, recall, f1, conf_matrix, per_class_precision
        FROM models_metrics
        WHERE df_hash=? AND model_type=? AND model=? AND target=?
        LIMIT 1
    ''', (df_hash, model_type, model_name, target))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            'accuracy': row[0],
            'precision': row[1],
            'recall': row[2],
            'f1_score': row[3],
            'conf_matrix': np.array(json.loads(row[4])),
            'per_class_precision': np.array(json.loads(row[5]))
        }
    return None

def display_cached_results(cached_metrics, model_name, label_encoder):
    st.write(f"📊 Using cached results (from previous run)")
    st.write(f"Accuracy: {cached_metrics['accuracy']*100:.2f}%")
    st.write(f"Precision: {cached_metrics['precision']*100:.2f}%")
    st.write(f"Recall: {cached_metrics['recall']*100:.2f}%")
    st.write(f"F1 Score: {cached_metrics['f1_score']*100:.2f}%")
    
    # Confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cached_metrics['conf_matrix'], annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name}")
    st.pyplot(fig)

# Hyperparameter selectors for ML models
def get_ml_model_params(model_name):
    params = {}
    with st.expander(f"⚙ {model_name} Hyperparameters"):
        if model_name == "Logistic Regression":
            params['C'] = st.slider("Inverse of regularization strength", 0.01, 10.0, 1.0, 0.01)
            params['max_iter'] = st.slider("Maximum iterations", 100, 1000, 100, 50)
            params['solver'] = st.selectbox("Solver", ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'])
            
        elif model_name == "Random Forest":
            params['n_estimators'] = st.slider("Number of trees", 10, 500, 100, 10)
            params['max_depth'] = st.slider("Max depth", 2, 50, 5, 1)
            params['min_samples_split'] = st.slider("Min samples split", 2, 20, 2, 1)
            
        elif model_name == "SVM":
            params['C'] = st.slider("Regularization parameter", 0.1, 10.0, 1.0, 0.1)
            params['kernel'] = st.selectbox("Kernel", ['linear', 'poly', 'rbf', 'sigmoid'])
            if params['kernel'] == 'poly':
                params['degree'] = st.slider("Polynomial degree", 2, 5, 3, 1)
                
        elif model_name == "KNN":
            params['n_neighbors'] = st.slider("Number of neighbors", 1, 50, 5, 1)
            params['weights'] = st.selectbox("Weight function", ['uniform', 'distance'])
            
        elif model_name == "Decision Tree":
            params['max_depth'] = st.slider("Max depth", 2, 50, 5, 1)
            params['min_samples_split'] = st.slider("Min samples split", 2, 20, 2, 1)
            params['criterion'] = st.selectbox("Split criterion", ['gini', 'entropy'])
            
        elif model_name == "Naïve Bayes":
            params['var_smoothing'] = st.slider("Smoothing parameter", 1e-9, 1e-1, 1e-9, format="%e")
            
    return params

# Dynamic DL layer configuration
def configure_dl_model(input_dim, num_classes):
    layers = []
    with st.expander("🧠 Deep Learning Architecture Configuration"):
        st.write("Configure each layer of your neural network:")
        
        # Input layer
        st.write(f"Input shape: {input_dim}")
        
        # Hidden layers
        num_layers = st.slider("Number of hidden layers", 1, 10, 2)
        
        for i in range(num_layers):
            cols = st.columns(3)
            with cols[0]:
                units = st.number_input(f"Layer {i+1} units", 1, 1024, 128 if i == 0 else 64, key=f"units_{i}")
            with cols[1]:
                activation = st.selectbox(
                    f"Layer {i+1} activation",
                    ['relu', 'sigmoid', 'tanh', 'elu', 'selu'],
                    key=f"act_{i}"
                )
            with cols[2]:
                dropout = st.slider(f"Layer {i+1} dropout", 0.0, 0.9, 0.3, 0.05, key=f"drop_{i}")
            
            layers.append({'units': units, 'activation': activation, 'dropout': dropout})
        
        # Output layer
        layers.append({'units': num_classes, 'activation': 'softmax', 'dropout': 0.0})
        
        # Training parameters
        cols = st.columns(3)
        with cols[0]:
            learning_rate = st.slider("Learning rate", 0.0001, 0.1, 0.001, 0.01, format="%f")
        with cols[1]:
            batch_size = st.slider("Batch size", 8, 256, 32, 16)
        with cols[2]:
            epochs = st.slider("Epochs", 10, 200, 20, 5)
    
    return layers, learning_rate, batch_size, epochs

def train_and_evaluate_ml_model(model_name, X_train, X_test, y_train, y_test, df, label_encoder):
    # Get model parameters
    params = get_ml_model_params(model_name)
    
    # Initialize model with selected parameters
    if model_name == "Logistic Regression":
        model = LogisticRegression(**params, random_state=42)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(**params, random_state=42)
    elif model_name == "SVM":
        model = SVC(**params, random_state=42)
    elif model_name == "KNN":
        model = KNeighborsClassifier(**params)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(**params, random_state=42)
    elif model_name == "Naïve Bayes":
        model = GaussianNB(**params)
    
    # Prepare data based on model requirements
    X_train_model = X_train.values if not issparse(X_train) else X_train.toarray()
    X_test_model = X_test.values if not issparse(X_test) else X_test.toarray()
    
    # Train model
    with st.spinner(f"Training {model_name}..."):
        model.fit(X_train_model, y_train)
        y_pred = model.predict(X_test_model)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    per_class_precision = precision_score(y_test, y_pred, average=None)
    
    metrics_dict = {
        'accuracy': accuracy,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1_score': report['weighted avg']['f1-score']
    }
    
    # Save to database
    save_metrics_to_db(df, "ML", model_name, metrics_dict, conf_matrix, per_class_precision, target_column)
    
    # Display metrics
    st.write(f"Accuracy: {accuracy*100:.2f}%")
    st.write(f"Precision: {metrics_dict['precision']*100:.2f}%")
    st.write(f"Recall: {metrics_dict['recall']*100:.2f}%")
    st.write(f"F1 Score: {metrics_dict['f1_score']*100:.2f}%")
    
    # Per-class metrics
    st.write("\n*Per-Class Metrics:*")
    metrics_df = pd.DataFrame({
        'Class': label_encoder.classes_,
        'Precision': precision_score(y_test, y_pred, average=None),
        'Recall': [report[str(i)]['recall'] for i in range(len(label_encoder.classes_))],
        'F1-Score': [report[str(i)]['f1-score'] for i in range(len(label_encoder.classes_))]
    })
    st.dataframe(metrics_df.style.format({
        'Precision': '{:.2%}',
        'Recall': '{:.2%}',
        'F1-Score': '{:.2%}'
    }))
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=range(len(label_encoder.classes_)))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name}")
    st.pyplot(fig)
    
    return accuracy

def train_and_evaluate_dl_model(X_train, X_test, y_train, y_test, y_train_cat, y_test_cat, 
                              df, label_encoder, input_dim, num_classes):
    # Configure the model
    layers, learning_rate, batch_size, epochs = configure_dl_model(input_dim, num_classes)
    
    # Build model
    model = Sequential()
    
    # Add first hidden layer with input_dim
    model.add(Dense(layers[0]['units'], input_dim=input_dim, activation=layers[0]['activation']))
    if layers[0]['dropout'] > 0:
        model.add(Dropout(layers[0]['dropout']))
    
    # Add remaining hidden layers
    for layer in layers[1:-1]:
        model.add(Dense(layer['units'], activation=layer['activation']))
        if layer['dropout'] > 0:
            model.add(Dropout(layer['dropout']))
    
    # Add output layer
    model.add(Dense(layers[-1]['units'], activation=layers[-1]['activation']))
    
    # Compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )
    
    # Train model
    with st.spinner("Training Deep Neural Network..."):
        history = model.fit(
            X_train, y_train_cat,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            validation_split=0.2
        )
    
    # Evaluate model
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    per_class_precision = precision_score(y_test, y_pred, average=None)
    
    metrics_dict = {
        'accuracy': accuracy,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1_score': report['weighted avg']['f1-score']
    }
    
    # Save to database
    model_name = f"DNN_{len(layers)-1}layers"
    save_metrics_to_db(df, "DL", model_name, metrics_dict, conf_matrix, per_class_precision, target_column)
    
    # Display metrics
    st.write(f"Accuracy: {accuracy*100:.2f}%")
    st.write(f"Precision: {metrics_dict['precision']*100:.2f}%")
    st.write(f"Recall: {metrics_dict['recall']*100:.2f}%")
    st.write(f"F1 Score: {metrics_dict['f1_score']*100:.2f}%")
    
    # Per-class metrics
    st.write("\n*Per-Class Metrics:*")
    metrics_df = pd.DataFrame({
        'Class': label_encoder.classes_,
        'Precision': precision_score(y_test, y_pred, average=None),
        'Recall': [report[str(i)]['recall'] for i in range(len(label_encoder.classes_))],
        'F1-Score': [report[str(i)]['f1-score'] for i in range(len(label_encoder.classes_))]
    })
    st.dataframe(metrics_df.style.format({
        'Precision': '{:.2%}',
        'Recall': '{:.2%}',
        'F1-Score': '{:.2%}'
    }))
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=range(len(label_encoder.classes_)))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - Deep Neural Network")
    st.pyplot(fig)
    
    # Plot training history
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history.history['accuracy'], label='Train Accuracy')
    ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax[0].set_title('Model Accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].legend()
    
    ax[1].plot(history.history['loss'], label='Train Loss')
    ax[1].plot(history.history['val_loss'], label='Validation Loss')
    ax[1].set_title('Model Loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].legend()
    
    st.pyplot(fig)
    
    return accuracy

def display_model_comparison(accuracy_scores):
    st.subheader("📈 Model Accuracy Comparison")
    comparison_df = pd.DataFrame.from_dict(accuracy_scores, orient='index', columns=['Accuracy'])
    st.bar_chart(comparison_df)

# Initialize database
init_db()

# File upload and data processing
uploaded_file = st.file_uploader("Upload your dataset (CSV only)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📄 Preview of Data", df.head())

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

    X = df[selected_features]
    y = df[target_column]

    # Encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)

    # Feature processing functions
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

    # Categorize columns
    text_cols = [col for col in X.columns if is_text_column(X[col])]
    cat_num_cols = [col for col in X.columns if is_categorical_numeric(X[col]) and col not in text_cols]
    pure_num_cols = [col for col in X.select_dtypes(include=np.number).columns 
                    if col not in cat_num_cols and col not in text_cols]

    final_features = []
    feature_info = []

    # Process text columns
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

    # Process numeric categorical columns
    if cat_num_cols:
        num_cat_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('scaler', MinMaxScaler())
        ])
        num_cat_data = num_cat_transformer.fit_transform(X[cat_num_cols])
        num_cat_df = pd.DataFrame(num_cat_data, columns=cat_num_cols)
        final_features.append(num_cat_df)
        feature_info.append(f"Numeric categorical (min-max scaled): {', '.join(cat_num_cols)}")

    # Process pure numeric columns
    if pure_num_cols:
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        numeric_data = numeric_transformer.fit_transform(X[pure_num_cols])
        numeric_df = pd.DataFrame(numeric_data, columns=pure_num_cols)
        final_features.append(numeric_df)
        feature_info.append(f"Continuous numeric (standard scaled): {', '.join(pure_num_cols)}")

    # Combine all features
    try:
        X_all_transformed_df = pd.concat(final_features, axis=1)
        numeric_cols = pure_num_cols + cat_num_cols
        
        # Display feature processing summary
        st.subheader("🔍 Feature Processing Summary")
        for info in feature_info:
            st.write(f"✅ {info}")
            
    except Exception as e:
        st.error(f"❌ Error combining features: {str(e)}")
        st.stop()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_all_transformed_df, y_encoded, test_size=0.2, random_state=42)
    
    # Prepare data for deep learning models
    X_train_dl = X_train.values if not issparse(X_train) else X_train.toarray()
    X_test_dl = X_test.values if not issparse(X_test) else X_test.toarray()
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

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
    tab1, tab2 = st.tabs([
        "🧠 Classical ML", 
        "🧬 Deep Learning"
    ])

    with tab1:  # Classical ML
        st.subheader("Classical Machine Learning Models")
        selected_ml_models = st.multiselect(
            "Select ML models to train",
            ml_models,
            default=["Random Forest", "SVM"]
        )
        
        for name in selected_ml_models:
            st.subheader(f"🤖 Model: {name}")
            
            cached_metrics = load_metrics_if_exist(df, "ML", name, target_column)
            
            if cached_metrics:
                display_cached_results(cached_metrics, name, label_encoder)
                accuracy_scores[name] = cached_metrics['accuracy'] * 100
            else:
                accuracy = train_and_evaluate_ml_model(
                    name, X_train, X_test, y_train, y_test, 
                    df, label_encoder
                )
                accuracy_scores[name] = accuracy * 100

        if accuracy_scores:
            display_model_comparison(accuracy_scores)

    with tab2:  # Deep Learning
        st.subheader("Deep Learning Model")
        
        cached_metrics = load_metrics_if_exist(df, "DL", "DNN", target_column)
        
        if cached_metrics:
            display_cached_results(cached_metrics, "Deep Neural Network", label_encoder)
            accuracy_scores["Deep Neural Network"] = cached_metrics['accuracy'] * 100
        else:
            accuracy = train_and_evaluate_dl_model(
                X_train_dl, X_test_dl, y_train, y_test, y_train_cat, y_test_cat,
                df, label_encoder, X_train_dl.shape[1], num_classes
            )
            accuracy_scores["Deep Neural Network"] = accuracy * 100

        if accuracy_scores:
<<<<<<< HEAD
            display_model_comparison(accuracy_scores)
=======
            display_model_comparison(accuracy_scores)
>>>>>>> 5b135de8f50bfb0569ade50420a3eab83228a65b
