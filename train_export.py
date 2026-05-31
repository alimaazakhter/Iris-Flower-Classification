import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

def train_and_export():
    csv_path = "Iris.csv"
    if not os.path.exists(csv_path):
        print(f"Error: Missing dataset '{csv_path}'. Cannot train.")
        return

    print("--- Iris Flower Classification Model Exporter ---")
    print(f"1. Loading dataset '{csv_path}'...")
    df = pd.read_csv(csv_path)
    
    # 2. Preprocess Data
    print("2. Preprocessing: Removing duplicates and encoding targets...")
    original_size = len(df)
    df.drop_duplicates(inplace=True)
    print(f"   Removed {original_size - len(df)} duplicate row(s).")
    
    le = LabelEncoder()
    df['Species_encoded'] = le.fit_transform(df['Species'])
    
    # Map species to integers
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"   Target Mapping: {mapping}")

    # Features and Target
    X = df.drop(['Id', 'Species', 'Species_encoded'], axis=1, errors='ignore')
    y = df['Species_encoded']
    
    # 3. Train-Test Split (80% Train, 20% Test)
    print("3. Splitting dataset (80% training, 20% testing)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train Model
    print("4. Fitting Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 5. Evaluate Model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   Model Accuracy on Test Split: {accuracy * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # 6. Save binary pickle files for FastAPI App
    print("5. Exporting model binary artifacts...")
    model_name = "iris_logistic_model.pkl"
    encoder_name = "label_encoder.pkl"
    joblib.dump(model, model_name)
    joblib.dump(le, encoder_name)
    print(f"   Saved '{model_name}'")
    print(f"   Saved '{encoder_name}'")

    # 7. Print JS Model parameters for client-side embedding reference
    print("\n--- JavaScript Coefficients Export (Reference) ---")
    print("If you wanted to embed the Scikit-Learn weights directly into JS:")
    print(f"Classes: {le.classes_.tolist()}")
    print(f"Features: {X.columns.tolist()}")
    print(f"Coefficients (W):\n{model.coef_.tolist()}")
    print(f"Intercepts (b):\n{model.intercept_.tolist()}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    train_and_export()
