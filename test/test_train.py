import os
import pandas as pd
from congen_ai.ai_main_csv import train_from_csv
from sklearn.datasets import load_iris

def generate_test_csv(path="test_data.csv"):
    iris = load_iris(as_frame=True)
    df = iris.frame
    df["target"] = iris.target
    df.to_csv(path, index=False)
    return path

def test_train_on_csv():
    csv_path = generate_test_csv()
    model, bag_acc, stack_acc = train_from_csv(csv_path, "target")
    
    assert bag_acc > 0.7, "Bagging accuracy too low!"
    assert stack_acc > 0.7, "Stacking accuracy too low!"
    assert hasattr(model, "predict"), "Model doesn't have predict method!"
    
    print("✅ Test passed: Ensemble model trained and evaluated correctly.")

if __name__ == "__main__":
    test_train_on_csv()
