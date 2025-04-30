import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
import numpy as np
import joblib
import os
from datetime import datetime

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Define available models from models directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
available_models = []

# Check which models are available
if os.path.exists(MODELS_DIR):
    for model_file in os.listdir(MODELS_DIR):
        if model_file.endswith('.joblib'):
            model_name = model_file.replace('_model.joblib', '').replace('_', ' ').title()
            available_models.append({
                'label': model_name,
                'value': model_file
            })

# Define the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("💳 Credit Card Fraud Detection", className="text-center my-4"),
            html.Hr(),
        ])
    ]),
    
    # File Upload Section
    dbc.Row([
        dbc.Col([
            html.H3("Step 1: Upload Your Data", className="mb-3"),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                multiple=False
            ),
            html.Div(id='output-data-upload'),
        ], width=12)
    ]),
    
    # Model Selection Section
    dbc.Row([
        dbc.Col([
            html.H3("Step 2: Select Model", className="mb-3"),
            dcc.Dropdown(
                id='model-selector',
                options=available_models,
                value=available_models[0]['value'] if available_models else None,
                style={'width': '100%'}
            ),
            html.Div(id='model-info', className="mt-2"),
        ], width=12)
    ], className="my-4"),
    
    # Analysis Button
    dbc.Row([
        dbc.Col([
            dbc.Button("Analyze Data", id='analyze-button', color="primary", className="mb-4"),
        ], width=12)
    ]),
    
    # Results Section
    dbc.Row([
        dbc.Col([
            html.H3("Results", className="mb-3"),
            html.Div(id='results-container'),
        ], width=12)
    ]),
    
    # Visualizations
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='fraud-distribution'),
        ], width=6),
        dbc.Col([
            dcc.Graph(id='amount-distribution'),
        ], width=6)
    ]),
    
    # Detailed Results
    dbc.Row([
        dbc.Col([
            html.H4("Detailed Analysis", className="mb-3"),
            html.Div(id='detailed-results'),
        ], width=12)
    ])
], fluid=True)

# Callback for file upload
@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    if contents is None:
        return html.Div("No file uploaded yet")
    
    return html.Div([
        html.H5(f"File uploaded: {filename}"),
        html.Hr()
    ])

# Callback for model info
@app.callback(
    Output('model-info', 'children'),
    Input('model-selector', 'value')
)
def update_model_info(selected_model):
    if not selected_model:
        return html.Div("No model selected")
    
    model_path = os.path.join(MODELS_DIR, selected_model)
    if not os.path.exists(model_path):
        return html.Div("Selected model not found", className="text-danger")
    
    try:
        model = joblib.load(model_path)
        return html.Div([
            html.P(f"Model Type: {type(model).__name__}"),
            html.P(f"Model Parameters: {model.get_params()}")
        ])
    except Exception as e:
        return html.Div(f"Error loading model: {str(e)}", className="text-danger")

# Callback for analysis
@app.callback(
    [Output('results-container', 'children'),
     Output('fraud-distribution', 'figure'),
     Output('amount-distribution', 'figure'),
     Output('detailed-results', 'children')],
    [Input('analyze-button', 'n_clicks')],
    [State('upload-data', 'contents'),
     State('model-selector', 'value')]
)
def analyze_data(n_clicks, contents, selected_model):
    if n_clicks is None or contents is None or not selected_model:
        raise PreventUpdate
    
    # Parse the uploaded file
    content_type, content_string = contents.split(',')
    import base64
    decoded = base64.b64decode(content_string)
    
    try:
        df = pd.read_csv(pd.io.common.BytesIO(decoded))
    except Exception as e:
        return html.Div([
            html.H5("Error reading file"),
            html.P(str(e))
        ]), {}, {}, html.Div()
    
    # Basic data validation
    required_columns = ['Time', 'Amount', 'Class']
    if not all(col in df.columns for col in required_columns):
        return html.Div([
            html.H5("Invalid data format"),
            html.P("The uploaded file must contain Time, Amount, and Class columns")
        ]), {}, {}, html.Div()
    
    # Load the selected model
    model_path = os.path.join(MODELS_DIR, selected_model)
    try:
        model = joblib.load(model_path)
    except Exception as e:
        return html.Div([
            html.H5("Error loading model"),
            html.P(str(e))
        ]), {}, {}, html.Div()
    
    # Prepare features and target
    X = df.drop(['Class', 'Time'], axis=1)
    y = df['Class']
    
    # Make predictions
    try:
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
    except Exception as e:
        return html.Div([
            html.H5("Error making predictions"),
            html.P(str(e))
        ]), {}, {}, html.Div()
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    f1 = f1_score(y, predictions)
    auc_roc = roc_auc_score(y, probabilities) if probabilities is not None else None
    
    # Create fraud distribution plot
    fraud_dist = px.pie(
        values=[len(df[df['Class'] == 0]), len(df[df['Class'] == 1])],
        names=['Legitimate', 'Fraudulent'],
        title='Transaction Distribution'
    )
    
    # Create amount distribution plot
    amount_dist = px.box(
        df,
        x='Class',
        y='Amount',
        title='Transaction Amount Distribution by Class',
        labels={'Class': 'Transaction Type', 'Amount': 'Amount ($)'}
    )
    
    # Prepare detailed results
    metrics_table = [
        html.Tr([html.Td("Accuracy"), html.Td(f"{accuracy:.4f}")]),
        html.Tr([html.Td("Precision"), html.Td(f"{precision:.4f}")]),
        html.Tr([html.Td("Recall"), html.Td(f"{recall:.4f}")]),
        html.Tr([html.Td("F1 Score"), html.Td(f"{f1:.4f}")])
    ]
    
    if auc_roc is not None:
        metrics_table.append(html.Tr([html.Td("AUC-ROC"), html.Td(f"{auc_roc:.4f}")]))
    
    detailed_results = dbc.Card([
        dbc.CardBody([
            html.H4("Model Performance Metrics"),
            html.Table(metrics_table, className="table table-striped"),
            
            html.H4("Fraud Summary"),
            html.P(f"Total Transactions: {len(df)}"),
            html.P(f"Fraudulent Transactions: {len(df[df['Class'] == 1])}"),
            html.P(f"Legitimate Transactions: {len(df[df['Class'] == 0])}"),
            
            html.H4("Feature Importance"),
            html.P("Feature importance analysis will be displayed here")
        ])
    ])
    
    return (
        html.Div([
            html.H4(f"Analysis Results using {selected_model.replace('_model.joblib', '').replace('_', ' ').title()}"),
            html.P(f"Model successfully analyzed {len(df)} transactions")
        ]),
        fraud_dist,
        amount_dist,
        detailed_results
    )

if __name__ == '__main__':
    app.run(debug=True) 