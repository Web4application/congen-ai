import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def train_from_csv(csv_path, target_column, output_model_path="stack_model.pkl"):
    # Load the CSV file
    df = pd.read_csv(csv_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in CSV.")

    # Prepare features and labels
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define base learners
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    knn_model = KNeighborsClassifier(n_neighbors=5)

    # Bagging ensemble
    bagging_model = BaggingClassifier(base_estimator=RandomForestClassifier(), n_estimators=10, random_state=42)
    bagging_model.fit(X_train, y_train)
    bagging_accuracy = accuracy_score(y_test, bagging_model.predict(X_test))

    # Stacking ensemble
    estimators = [('rf', rf_model), ('knn', knn_model)]
    stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5)
    stacking_model.fit(X_train, y_train)
    stacking_accuracy = accuracy_score(y_test, stacking_model.predict(X_test))

    # Save the stacking model
    joblib.dump(stacking_model, output_model_path)

    # Output results
    print(f"✅ Bagging Accuracy: {bagging_accuracy * 100:.2f}%")
    print(f"✅ Stacking Accuracy: {stacking_accuracy * 100:.2f}%")

    return stacking_model, bagging_accuracy, stacking_accuracy

# Command-line interface
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train ensemble models from CSV")
    parser.add_argument("--csv", required=True, help="Path to your CSV file")
    parser.add_argument("--target", required=True, help="Name of the target column")
    args = parser.parse_args()

    train_from_csv(args.csv, args.target)
