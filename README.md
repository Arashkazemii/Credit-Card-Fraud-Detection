# 💳 Credit Card Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ML](https://img.shields.io/badge/Machine%20Learning-XGBoost%2C%20LightGBM-orange)](https://xgboost.ai/)
[![Dashboard](https://img.shields.io/badge/Dashboard-Dash-purple)](https://dash.plotly.com/)
[![Code Quality](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A powerful machine learning project that implements various classification algorithms to detect fraudulent credit card transactions. The project includes an interactive web dashboard for real-time fraud detection and analysis.

## 📋 Table of Contents

- [✨ Features](#-features)
- [🎯 Project Overview](#-project-overview)
- [📊 Dataset Information](#-dataset-information)
- [🤖 Machine Learning Models](#-machine-learning-models)
- [📁 Project Structure](#-project-structure)
- [🛠️ Installation](#-installation)
- [🚀 Usage](#-usage)
- [📈 Performance Metrics](#-performance-metrics)
- [🤝 Contributing](#-contributing)
- [📝 License](#-license)

## ✨ Features

- 🚀 Multiple ML models (XGBoost, LightGBM, Random Forest)
- 📊 Interactive web dashboard using Dash
- ⚡ Real-time fraud detection
- 📈 Data visualization with Plotly
- ⚖️ Handling of imbalanced data
- 🔍 Feature importance analysis
- 📱 Responsive design for all devices
- 🔒 Secure data handling
- 📊 Comprehensive performance metrics

## 🎯 Project Overview

This project provides a user-friendly web interface for credit card fraud detection. Users can:

1. Upload their credit card transaction data
2. Select from available pre-trained models
3. View real-time fraud detection results
4. Analyze transaction patterns through interactive visualizations
5. Get detailed performance metrics and fraud statistics

## 📊 Dataset Information

The application expects credit card transaction data in CSV format with the following columns:

- `Time`: Time of the transaction
- `V1-V28`: Anonymized features (PCA transformed)
- `Amount`: Transaction amount
- `Class`: Target variable (0: legitimate, 1: fraudulent)

### Sample Data Format
```csv
Time,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount,Class
0,-1.359807,-0.072781,2.536347,1.378155,-0.338321,0.462388,0.239599,0.098698,0.363787,0.090794,-0.551600,-0.617801,-0.991390,-0.311169,1.468177,-0.470401,0.207971,0.025791,0.403993,0.251412,-0.018307,0.277838,-0.110474,0.066928,0.128539,-0.189115,0.133558,-0.021053,149.62,0
```

## 🤖 Machine Learning Models

The project supports multiple machine learning models:

1. **Random Forest**
   - Robust to outliers
   - Feature selection capabilities
   - Pre-trained model available

2. **XGBoost** (Upcoming)
   - Optimized for handling imbalanced data
   - Feature importance analysis
   - High prediction accuracy

3. **LightGBM** (Upcoming)
   - Fast training and prediction
   - Memory efficient
   - Gradient-based one-side sampling

## 📁 Project Structure

```
credit-card-fraud-detection/
├── 📂 data/               # Dataset files
├── 📂 models/            # Trained model files
│   ├── random_forest_model.joblib
│   ├── xgboost_model.joblib (Upcoming)
│   └── lightgbm_model.joblib (Upcoming)
├── 📂 src/               # Source code
│   ├── app.py           # Web application
│   └── utils/           # Utility functions
├── 📄 docker-compose.yml   # Docker-Compose 
├── 📄 Dockerfile   # Dockerfile
├── 📄 requirements.txt   # Project dependencies
└── 📄 LICENSE           # MIT License
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git
- Docker and Docker Compose (for containerized deployment)

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/Arashkazemii/Credit-Card-Fraud-Detection
cd Credit-Card-Fraud-Detection
```

2. Choose one of the following installation methods:

#### Method 1: Traditional Installation
Create and activate a virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/MacOS
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

#### Method 2: Docker Installation
Build and run the application using Docker Compose:
```bash
docker-compose up --build
```

The application will be available at `http://localhost:8050`

## 🚀 Usage

### Running the Web Application

1. Start the web dashboard:
```bash
python src/app.py
```

2. Access the dashboard at `http://localhost:8050`

### Using the Application

1. **Upload Data**
   - Click the upload area or drag and drop your CSV file
   - Ensure your data follows the required format

2. **Select Model**
   - Choose from available pre-trained models
   - View model information and parameters

3. **Analyze Data**
   - Click "Analyze Data" to process your transactions
   - View fraud detection results and visualizations

4. **View Results**
   - Transaction distribution (pie chart)
   - Amount distribution by class (box plot)
   - Performance metrics (accuracy, precision, recall, F1 score)
   - Fraud summary statistics

## 📈 Performance Metrics

The models are evaluated using the following metrics:

- Accuracy: 99.9%
- Precision: 0.92
- Recall: 0.85
- F1-Score: 0.88
- AUC-ROC: 0.98

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Write clear commit messages
- Add tests for new features
- Update documentation as needed
- Use type hints for better code clarity

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <sub>Built with ❤️ by <a href="https://github.com/Arashkazemii">Arash Kazemi</a></sub>
</div>