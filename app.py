import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="🌲 Random Forest Classifier", layout="centered")
st.title("🌲 Random Forest Classifier (Streamlit)")

# ----------------------------
# Upload CSV
# ----------------------------
uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ----------------------------
    # Select target column
    # ----------------------------
    target_column = st.selectbox("Select Target Column", df.columns)

    if target_column:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Encode categorical features
        X = pd.get_dummies(X)

        # Encode target if categorical
        label_encoder = None
        if y.dtype == "object":
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

        # Split dataset
        test_size = st.slider("Test size (%)", 10, 50, 20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=42
        )

        # Train Random Forest
        n_estimators = st.number_input(
            "Number of trees (n_estimators)",
            min_value=10, max_value=500, value=100, step=10
        )

        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.success(f"✅ Model Accuracy: {acc:.2f}")

        # Confusion matrix
        if st.checkbox("Show Confusion Matrix"):
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        # Save model safely
        model_file = "rf_model.pkl"
        if os.path.exists(model_file):
            st.warning(f"Model file '{model_file}' already exists. It will be overwritten.")
        joblib.dump(model, model_file)
        st.info(f"Model saved as '{model_file}'")

        # ----------------------------
        # Make Prediction
        # ----------------------------
        st.subheader("🔮 Make a Prediction")

        input_data = []
        for col in X.columns:
            val = st.number_input(f"{col}", value=0.0)
            input_data.append(val)

        if st.button("Predict"):
            input_array = np.array(input_data).reshape(1, -1)
            try:
                prediction = model.predict(input_array)
                if label_encoder:
                    prediction = label_encoder.inverse_transform(prediction)
                st.success(f"Prediction: {prediction[0]}")

                # Optional: show probability
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(input_array)[0][1]
                    st.info(f"Prediction probability for class 1: {prob:.2f}")

            except ValueError as e:
                st.error(f"❌ Prediction failed: {e}")

