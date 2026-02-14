import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                             roc_auc_score, precision_score, recall_score, f1_score, 
                             matthews_corrcoef)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

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
    model_options = [
        "Logistic Regression",
        "Decision Tree Classifier",
        "K-Nearest Neighbor",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
    
    model_option = st.selectbox(
        "Select Model",
        model_options
    )

    if model_option == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)

    elif model_option == "Decision Tree Classifier":
        model = DecisionTreeClassifier(random_state=42)

    elif model_option == "K-Nearest Neighbor":
        model = KNeighborsClassifier(n_neighbors=5)

    elif model_option == "Naive Bayes":
        model = GaussianNB()

    elif model_option == "Random Forest":
        model = RandomForestClassifier(random_state=42)

    elif model_option == "XGBoost":
        if XGBClassifier is None:
            st.error("XGBoost is not installed. Please install it using: pip install xgboost")
            st.stop()
        model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

    if st.button("Train Model"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ========== C. Evaluation Metrics ==========
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # Calculate AUC score (for binary and multi-class)
        auc = None
        try:
            n_classes = len(np.unique(y_test))
            
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                if n_classes == 2:
                    # Binary classification: use probability of positive class
                    auc = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    # Multi-class classification
                    auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted', labels=np.unique(y_test))
            elif hasattr(model, 'decision_function'):
                # For models like SVM that have decision_function
                y_scores = model.decision_function(X_test)
                if n_classes == 2:
                    auc = roc_auc_score(y_test, y_scores)
                else:
                    auc = roc_auc_score(y_test, y_scores, multi_class='ovr', average='weighted')
            else:
                auc = None
        except Exception as e:
            print(f"AUC Calculation Error: {str(e)}")
            auc = None

        st.subheader("Evaluation Metrics")
        
        # Display metrics in a formatted table
        metrics_data = {
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC Score", "MCC Score"],
            "Value": [
                f"{accuracy:.4f}",
                f"{precision:.4f}",
                f"{recall:.4f}",
                f"{f1:.4f}",
                f"{auc:.4f}" if auc is not None else "N/A",
                f"{mcc:.4f}"
            ]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        st.table(metrics_df)
        
        # Display individual metrics with larger text
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{accuracy:.4f}")
        with col2:
            st.metric("Precision", f"{precision:.4f}")
        with col3:
            st.metric("Recall", f"{recall:.4f}")
            
        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric("F1 Score", f"{f1:.4f}")
        with col5:
            st.metric("MCC Score", f"{mcc:.4f}")
        with col6:
            if auc is not None:
                st.metric("AUC Score", f"{auc:.4f}")
            else:
                st.metric("AUC Score", "N/A")

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
