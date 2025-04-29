import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_class_distribution(y):
    """
    Plot the distribution of classes in the dataset.
    
    Args:
        y (numpy.ndarray): Target labels
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()

def plot_feature_distribution(df, feature):
    """
    Plot the distribution of a specific feature.
    
    Args:
        df (pandas.DataFrame): Dataset
        feature (str): Name of the feature to plot
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=feature, hue='Class', multiple='stack')
    plt.title(f'Distribution of {feature} by Class')
    plt.show()

def calculate_class_weights(y):
    """
    Calculate class weights for imbalanced dataset.
    
    Args:
        y (numpy.ndarray): Target labels
        
    Returns:
        dict: Class weights
    """
    unique_classes = np.unique(y)
    class_weights = {}
    
    for cls in unique_classes:
        class_weights[cls] = len(y) / (len(unique_classes) * np.sum(y == cls))
    
    return class_weights

def save_metrics(metrics, file_path):
    """
    Save evaluation metrics to a file.
    
    Args:
        metrics (dict): Dictionary containing evaluation metrics
        file_path (str): Path to save the metrics
    """
    with open(file_path, 'w') as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value}\n")
    print(f"Metrics saved to {file_path}") 