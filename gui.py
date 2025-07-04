import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import tempfile
import numpy as np

st.set_page_config(page_title="Congen‑AI Trainer", layout="centered")
st.title("🧠 Congen‑AI – CSV Ensemble Trainer")

uploaded = st.file_uploader("📂 Upload CSV", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("Preview")
    st.dataframe(df.head())

    target = st.selectbox("🎯 Target column", df.columns)
    if st.button("🚀 Train"):
        with st.spinner("Training…"):
            X = df.drop(columns=[target])
            y = df[target]
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            knn = KNeighborsClassifier(n_neighbors=5)

            bag = BaggingClassifier(base_estimator=RandomForestClassifier(),
                                    n_estimators=10, random_state=42)
            bag.fit(X_tr, y_tr)
            bag_acc = accuracy_score(y_te, bag.predict(X_te))

            stack = StackingClassifier(
                estimators=[('rf', rf), ('knn', knn)],
                final_estimator=LogisticRegression(),
                cv=5
            )
            stack.fit(X_tr, y_tr)
            stack_pred = stack.predict(X_te)
            stack_acc = accuracy_score(y_te, stack_pred)

            # Confusion‑matrix plot
            cm = confusion_matrix(y_te, stack_pred)
            fig, ax = plt.subplots()
            ax.set_title("Confusion Matrix (Stacking)")
            im = ax.imshow(cm)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j], ha="center", va="center")
            st.pyplot(fig)

            st.success(f"Bagging accuracy  : {bag_acc*100:.2f}%")
            st.success(f"Stacking accuracy : {stack_acc*100:.2f}%")

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
            joblib.dump(stack, tmp.name)
            st.download_button("📥 Download model", open(tmp.name, "rb"),
                               file_name="stack_model.pkl")
