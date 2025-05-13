import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def display_model_comparison(accuracy_scores):
    st.subheader("📈 Model Accuracy Comparison")
    comparison_df = pd.DataFrame.from_dict(accuracy_scores, orient='index', columns=['Accuracy'])
    st.bar_chart(comparison_df)