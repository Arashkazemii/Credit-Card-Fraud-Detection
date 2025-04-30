import numpy as np
import joblib
import pandas as pd
from preprocessing.preprocess import preprocess_data

def load_model(model_path):
    """
    Load a trained model from disk.
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        Trained model
    """
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def make_predictions(model, data):
    """
    Make predictions using the trained model.
    
    Args:
        model: Trained model
        data (pandas.DataFrame): Data to make predictions on
        
    Returns:
        numpy.ndarray: Predicted labels
    """
    try:
        # Preprocess the data (assuming it's in the same format as training data)
        X, _, _, _ = preprocess_data(data)
        
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        return predictions, probabilities
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None, None

if __name__ == "__main__":
    # Load the trained model
    model_path = "./models/random_forest_model.joblib"
    model = load_model(model_path)
    
    if model is not None:
        # Load new data (you can replace this with your new data)
        new_data_path = "./data/creditcard.csv"  # Replace with your new data path
        try:
            new_data = pd.read_csv(new_data_path)
            print(f"Loaded {len(new_data)} samples for prediction")
            
            # Make predictions
            predictions, probabilities = make_predictions(model, new_data)
            
            if predictions is not None:
                # Create a DataFrame with predictions and probabilities
                results = pd.DataFrame({
                    'Predicted_Class': predictions,
                    'Probability_Class_0': probabilities[:, 0],
                    'Probability_Class_1': probabilities[:, 1]
                })
                
                # Save predictions to CSV
                output_path = "./predictions/predictions.csv"
                results.to_csv(output_path, index=False)
                print(f"Predictions saved to {output_path}")
                
                # Print some statistics
                fraud_count = np.sum(predictions == 1)
                print(f"\nPrediction Summary:")
                print(f"Total samples: {len(predictions)}")
                print(f"Predicted fraud cases: {fraud_count}")
                print(f"Predicted non-fraud cases: {len(predictions) - fraud_count}")
                
        except Exception as e:
            print(f"Error loading data: {e}") 