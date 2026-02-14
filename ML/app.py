import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

st.title("Machine Learning Model Deployment App")

st.write("Upload your CSV dataset (small test dataset recommended).")

# ========== A. Dataset Upload ==========
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(data.head())

    target_column = st.selectbox("Select Target Column", data.columns)

    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Validate and convert target column for classification
    # Remove rows with NaN values in target
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask]

    # Convert to numeric if not already
    if y.dtype == 'object' or y.dtype == 'string':
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ========== B. Model Selection Dropdown ==========
    model_option = st.selectbox(
        "Select Model",
        ("Logistic Regression", "Random Forest", "SVM")
    )

    if model_option == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)

    elif model_option == "Random Forest":
        model = RandomForestClassifier()

    elif model_option == "SVM":
        model = SVC()

    if st.button("Train Model"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ========== C. Evaluation Metrics ==========
        accuracy = accuracy_score(y_test, y_pred)

        st.subheader("Evaluation Metrics")
        st.write(f"Accuracy: {accuracy:.4f}")

        # ========== D. Confusion Matrix ==========
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        # ========== Classification Report ==========
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred)
        st.text(report)

else:
    st.info("Please upload a CSV file to proceed.")
