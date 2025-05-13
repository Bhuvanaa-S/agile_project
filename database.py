import sqlite3
import json
import numpy as np
import pandas as pd
import hashlib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

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
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_params (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_metrics_id INTEGER,
            params TEXT,
            FOREIGN KEY (model_metrics_id) REFERENCES models_metrics(id)
        )
    ''')
    conn.commit()
    conn.close()

def save_metrics_to_db(df_original, model_type, model_name, metrics_dict, conf_matrix, per_class_precision, target, params=None):
    df_hash = get_df_hash(df_original)
    conn = sqlite3.connect("ml_results.db")
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO models_metrics (
            df_hash, model_type, model, accuracy, precision, recall, f1, conf_matrix, 
            per_class_precision, target
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        df_hash, model_type, model_name,
        metrics_dict['accuracy'], metrics_dict['precision'],
        metrics_dict['recall'], metrics_dict['f1_score'],
        json.dumps(conf_matrix.tolist()),
        json.dumps(per_class_precision.tolist()),
        target
    ))
    
    model_metrics_id = cursor.lastrowid
    
    if params:
        cursor.execute('''
            INSERT INTO model_params (model_metrics_id, params) VALUES (?, ?)
        ''', (model_metrics_id, json.dumps(params)))
    
    conn.commit()
    conn.close()

def load_metrics_if_exist(df_original, model_type, model_name, target, current_params=None):
    df_hash = get_df_hash(df_original)
    conn = sqlite3.connect("ml_results.db")
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT m.id, m.accuracy, m.precision, m.recall, m.f1, m.conf_matrix, m.per_class_precision, p.params
        FROM models_metrics m
        LEFT JOIN model_params p ON m.id = p.model_metrics_id
        WHERE m.df_hash=? AND m.model_type=? AND m.model=? AND m.target=?
        ORDER BY m.id DESC LIMIT 1
    ''', (df_hash, model_type, model_name, target))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        metrics_id, accuracy, precision, recall, f1, conf_matrix, per_class_precision, params_json = row
        saved_params = json.loads(params_json) if params_json else {}
        
        if current_params is None or saved_params == current_params:
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'conf_matrix': np.array(json.loads(conf_matrix)),
                'per_class_precision': np.array(json.loads(per_class_precision)),
                'params': saved_params
            }
    return None

def display_cached_results(cached_metrics, model_name, label_encoder):
    st.write(f"📊 Using cached results (from previous run)")
    if cached_metrics.get('params'):
        with st.expander("⚙️ Parameters used for these results"):
            st.json(cached_metrics['params'])
    st.write(f"*Accuracy:* {cached_metrics['accuracy']*100:.2f}%")
    st.write(f"*Precision:* {cached_metrics['precision']*100:.2f}%")
    st.write(f"*Recall:* {cached_metrics['recall']*100:.2f}%")
    st.write(f"*F1 Score:* {cached_metrics['f1_score']*100:.2f}%")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cached_metrics['conf_matrix'], annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name}")
    st.pyplot(fig)