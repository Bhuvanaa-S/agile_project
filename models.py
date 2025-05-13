import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from scipy.sparse import issparse
from database import save_metrics_to_db, load_metrics_if_exist, display_cached_results

def get_ml_model_params(model_name):
    params = {}
    with st.expander(f"⚙️ {model_name} Hyperparameters"):
        if model_name == "Logistic Regression":
            params['C'] = st.slider("Inverse of regularization strength", 0.01, 10.0, 1.0, 0.01, key=f"logreg_C_{model_name}")
            params['max_iter'] = st.slider("Maximum iterations", 100, 1000, 100, 50, key=f"logreg_max_iter_{model_name}")
            params['solver'] = st.selectbox("Solver", ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'], key=f"logreg_solver_{model_name}")
            
        elif model_name == "Random Forest":
            params['n_estimators'] = st.slider("Number of trees", 10, 500, 100, 10, key=f"rf_n_estimators_{model_name}")
            params['max_depth'] = st.slider("Max depth", 2, 50, 5, 1, key=f"rf_max_depth_{model_name}")
            params['min_samples_split'] = st.slider("Min samples split", 2, 20, 2, 1, key=f"rf_min_samples_split_{model_name}")
            
        elif model_name == "SVM":
            params['C'] = st.slider("Regularization parameter", 0.1, 10.0, 1.0, 0.1, key=f"svm_C_{model_name}")
            params['kernel'] = st.selectbox("Kernel", ['linear', 'poly', 'rbf', 'sigmoid'], key=f"svm_kernel_{model_name}")
            if params['kernel'] == 'poly':
                params['degree'] = st.slider("Polynomial degree", 2, 5, 3, 1, key=f"svm_degree_{model_name}")
                
        elif model_name == "KNN":
            params['n_neighbors'] = st.slider("Number of neighbors", 1, 50, 5, 1, key=f"knn_n_neighbors_{model_name}")
            params['weights'] = st.selectbox("Weight function", ['uniform', 'distance'], key=f"knn_weights_{model_name}")
            
        elif model_name == "Decision Tree":
            params['max_depth'] = st.slider("Max depth", 2, 50, 5, 1, key=f"dt_max_depth_{model_name}")
            params['min_samples_split'] = st.slider("Min samples split", 2, 20, 2, 1, key=f"dt_min_samples_split_{model_name}")
            params['criterion'] = st.selectbox("Split criterion", ['gini', 'entropy'], key=f"dt_criterion_{model_name}")
            
        elif model_name == "Naïve Bayes":
            params['var_smoothing'] = st.slider("Smoothing parameter", 1e-9, 1e-1, 1e-9, format="%e", key=f"nb_var_smoothing_{model_name}")
            
    return params

def configure_dl_model(input_dim, num_classes):
    layers = []
    with st.expander("🧠 Deep Learning Architecture Configuration"):
        st.write(f"Input shape: {input_dim}")
        num_layers = st.slider("Number of hidden layers", 1, 10, 2, key="dl_num_layers")
        
        for i in range(num_layers):
            cols = st.columns(3)
            with cols[0]:
                units = st.number_input(f"Layer {i+1} units", 1, 1024, 128 if i == 0 else 64, key=f"dl_units_{i}")
            with cols[1]:
                activation = st.selectbox(
                    f"Layer {i+1} activation",
                    ['relu', 'sigmoid', 'tanh', 'elu', 'selu'],
                    key=f"dl_act_{i}"
                )
            with cols[2]:
                dropout = st.slider(f"Layer {i+1} dropout", 0.0, 0.9, 0.3, 0.05, key=f"dl_drop_{i}")
            
            layers.append({'units': units, 'activation': activation, 'dropout': dropout})
        
        layers.append({'units': num_classes, 'activation': 'softmax', 'dropout': 0.0})
        
        cols = st.columns(3)
        with cols[0]:
            learning_rate = st.slider("Learning rate", 0.0001, 0.1, 0.001, 0.01, format="%f", key="dl_learning_rate")
        with cols[1]:
            batch_size = st.slider("Batch size", 8, 256, 32, 16, key="dl_batch_size")
        with cols[2]:
            epochs = st.slider("Epochs", 10, 200, 20, 5, key="dl_epochs")
    
    return layers, learning_rate, batch_size, epochs, {
        'layers': layers,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs
    }

def train_and_evaluate_ml_model(model_name, X_train, X_test, y_train, y_test, df, label_encoder, target_column):
    params = get_ml_model_params(model_name)
    cached_metrics = load_metrics_if_exist(df, "ML", model_name, target_column, params)
    
    if cached_metrics:
        display_cached_results(cached_metrics, model_name, label_encoder)
        return cached_metrics['accuracy']
    
    model_classes = {
        "Logistic Regression": LogisticRegression,
        "Random Forest": RandomForestClassifier,
        "SVM": SVC,
        "KNN": KNeighborsClassifier,
        "Decision Tree": DecisionTreeClassifier,
        "Naïve Bayes": GaussianNB
    }
    
    model = model_classes[model_name](**params, )
    X_train_model = X_train.values if not issparse(X_train) else X_train.toarray()
    X_test_model = X_test.values if not issparse(X_test) else X_test.toarray()
    
    with st.spinner(f"Training {model_name}..."):
        model.fit(X_train_model, y_train)
        y_pred = model.predict(X_test_model)
    
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
    
    save_metrics_to_db(df, "ML", model_name, metrics_dict, conf_matrix, per_class_precision, target_column, params)
    
    st.write(f"*Accuracy:* {accuracy*100:.2f}%")
    st.write(f"*Precision:* {metrics_dict['precision']*100:.2f}%")
    st.write(f"*Recall:* {metrics_dict['recall']*100:.2f}%")
    st.write(f"*F1 Score:* {metrics_dict['f1_score']*100:.2f}%")
    
    class_labels = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    metrics_df = pd.DataFrame({
        'Class': label_encoder.classes_,
        'Precision': [report[label]['precision'] for label in class_labels],
        'Recall': [report[label]['recall'] for label in class_labels],
        'F1-Score': [report[label]['f1-score'] for label in class_labels]
    })
    st.dataframe(metrics_df.style.format({
        'Precision': '{:.2%}', 'Recall': '{:.2%}', 'F1-Score': '{:.2%}'
    }))
    
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
                              df, label_encoder, target_column, input_dim, num_classes):
    layers, lr, batch_size, epochs, params = configure_dl_model(input_dim, num_classes)
    cached_metrics = load_metrics_if_exist(df, "DL", "DNN", target_column, params)
    
    if cached_metrics:
        display_cached_results(cached_metrics, "Deep Neural Network", label_encoder)
        return cached_metrics['accuracy']
    
    model = Sequential()
    model.add(Dense(layers[0]['units'], input_dim=input_dim, activation=layers[0]['activation'], 
                    name=f"dense_input_{layers[0]['units']}"))
    if layers[0]['dropout'] > 0:
        model.add(Dropout(layers[0]['dropout']))
    
    for i, layer in enumerate(layers[1:-1]):
        model.add(Dense(layer['units'], activation=layer['activation'], 
                        name=f"dense_{i+1}_{layer['units']}"))
        if layer['dropout'] > 0:
            model.add(Dropout(layer['dropout'], name=f"dropout_{i+1}_{layer['dropout']}"))
    
    model.add(Dense(layers[-1]['units'], activation=layers[-1]['activation'],
                    name=f"dense_output_{layers[-1]['units']}"))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
    
    with st.spinner("Training Deep Neural Network..."):
        history = model.fit(X_train, y_train_cat, epochs=epochs, batch_size=batch_size, verbose=0, validation_split=0.2)
    
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
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
    
    model_name = f"DNN_{len(layers)-1}layers"
    save_metrics_to_db(df, "DL", model_name, metrics_dict, conf_matrix, per_class_precision, target_column, params)
    
    st.write(f"*Accuracy:* {accuracy*100:.2f}%")
    st.write(f"*Precision:* {metrics_dict['precision']*100:.2f}%")
    st.write(f"*Recall:* {metrics_dict['recall']*100:.2f}%")
    st.write(f"*F1 Score:* {metrics_dict['f1_score']*100:.2f}%")
    
    class_labels = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    metrics_df = pd.DataFrame({
        'Class': label_encoder.classes_,
        'Precision': [report[label]['precision'] for label in class_labels],
        'Recall': [report[label]['recall'] for label in class_labels],
        'F1-Score': [report[label]['f1-score'] for label in class_labels]
    })
    st.dataframe(metrics_df.style.format({
        'Precision': '{:.2%}', 'Recall': '{:.2%}', 'F1-Score': '{:.2%}'
    }))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - Deep Neural Network")
    st.pyplot(fig)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    
    st.pyplot(fig)
    return accuracy