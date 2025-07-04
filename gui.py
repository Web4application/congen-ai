import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
import tempfile
import os

st.set_page_config(page_title="Congen-AI Trainer", layout="centered")

st.title("🧠 Congen-AI: Train on Your CSV")
st.write("Upload your dataset and train Bagging & Stacking models directly.")

uploaded_file = st.file_uploader("📁 Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of your data:")
    st.dataframe(df.head())

    target_column = st.selectbox("🎯 Select Target Column", df.columns)

    if st.button("🚀 Train Models"):
        with st.spinner("Training models..."):
            try:
                # Feature/Label split
                X = df.drop(columns=[target_column])
                y = df[target_column]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                knn_model = KNeighborsClassifier(n_neighbors=5)

                # Bagging
                bagging_model = BaggingClassifier(
                    base_estimator=RandomForestClassifier(),
                    n_estimators=10,
                    random_state=42
                )
                bagging_model.fit(X_train, y_train)
                bag_acc = accuracy_score(y_test, bagging_model.predict(X_test))

                # Stacking
                estimators = [('rf', rf_model), ('knn', knn_model)]
                stacking_model = StackingClassifier(
                    estimators=estimators,
                    final_estimator=LogisticRegression(),
                    cv=5
                )
                stacking_model.fit(X_train, y_train)
                stack_acc = accuracy_score(y_test, stacking_model.predict(X_test))

                # Save stacking model to temp file
                temp_model = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
                joblib.dump(stacking_model, temp_model.name)

                st.success(f"✅ Bagging Accuracy: {bag_acc * 100:.2f}%")
                st.success(f"✅ Stacking Accuracy: {stack_acc * 100:.2f}%")
                st.download_button("📥 Download Trained Model", open(temp_model.name, "rb"), file_name="stack_model.pkl")

            except Exception as e:
                st.error(f"❌ Error: {e}")
