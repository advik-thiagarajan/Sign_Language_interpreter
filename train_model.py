import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os

def train_isl_logic():
    csv_file = 'Indian Sign Language Gesture Landmarks.csv' 
    
    if not os.path.exists(csv_file):
        print(f"❌ Error: {csv_file} not found.")
        return

    print("📂 Loading and Balancing Dataset...")
    df = pd.read_csv(csv_file) 

    # Handle Label Column
    if 'label' in df.columns:
        X = df.drop('label', axis=1)
        y = df['label']
    else:
        X = df.iloc[:, 1:]
        y = df.iloc[:, 0]

    # STRENGTHEN THE MODEL: Use more trees and balance the classes
    print(f"🧠 Training Optimized Forest (150 trees)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y)
    
    # class_weight='balanced' prevents the model from favoring one sign (like '16')
    model = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    # Save both the model AND the feature count for the backend to use
    model_data = {
        'model': model,
        'features': X.shape[1]
    }

    with open('isl_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✅ SUCCESS! Test Accuracy: {round(model.score(X_test, y_test) * 100, 2)}%")

if __name__ == "__main__":
    train_isl_logic()