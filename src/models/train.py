import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

def train_model(X_train, y_train):
    """
    Train a Random Forest classifier on the preprocessed data.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training labels
        
    Returns:
        sklearn.ensemble.RandomForestClassifier: Trained model
    """
    # Initialize and train the model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set.
    
    Args:
        model: Trained model
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): Test labels
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Print evaluation metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def save_model(model, model_path):
    """
    Save the trained model to disk.
    
    Args:
        model: Trained model
        model_path (str): Path to save the model
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # Import preprocessing functions
    import sys
    sys.path.append('./src')
    from preprocessing.preprocess import load_data, preprocess_data
    
    # Load and preprocess data
    data_path = "./data/creditcard.csv"
    df = load_data(data_path)
    
    if df is not None:
        X_train, X_test, y_train, y_test = preprocess_data(df)
        
        # Train the model
        model = train_model(X_train, y_train)
        
        # Evaluate the model
        evaluate_model(model, X_test, y_test)
        
        # Save the model
        model_path = "./models/random_forest_model.joblib"
        save_model(model, model_path) 