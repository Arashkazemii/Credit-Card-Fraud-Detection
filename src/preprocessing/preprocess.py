import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Load the credit card transaction dataset.
    
    Args:
        file_path (str): Path to the CSV file containing the dataset
        
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

def preprocess_data(df):
    """
    Preprocess the credit card transaction data.
    
    Args:
        df (pandas.DataFrame): Raw dataset
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Example usage
    data_path = "./data/creditcard.csv"  # Update this path according to your dataset location
    df = load_data(data_path)
    
    if df is not None:
        X_train, X_test, y_train, y_test = preprocess_data(df)
        print("Data preprocessing completed successfully!")
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}") 