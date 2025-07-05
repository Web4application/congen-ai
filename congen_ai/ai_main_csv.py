
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def train_from_csv(csv_path, target_column, output_model_path="stack_model.pkl"):
    """Train Bagging & Stacking ensemble on any CSV."""
    df = pd.read_csv(csv_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in CSV.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)

    bag = BaggingClassifier(
        base_estimator=RandomForestClassifier(),
        n_estimators=10,
        random_state=42,
    ).fit(X_train, y_train)
    bag_acc = accuracy_score(y_test, bag.predict(X_test))

    stack = StackingClassifier(
        estimators=[('rf', rf), ('knn', knn)],
        final_estimator=LogisticRegression(),
        cv=5,
    ).fit(X_train, y_train)
    stack_acc = accuracy_score(y_test, stack.predict(X_test))

    joblib.dump(stack, output_model_path)

    print(f"Bagging accuracy : {bag_acc*100:.2f}%")
    print(f"Stacking accuracy: {stack_acc*100:.2f}%")

    return stack, bag_acc, stack_acc

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train ensembles from CSV.")
    parser.add_argument("--csv", required=True, help="CSV path")
    parser.add_argument("--target", required=True, help="Target column")
    args = parser.parse_args()
    train_from_csv(args.csv, args.target)
