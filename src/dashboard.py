import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Initialize the Dash app
app = dash.Dash(__name__)

# Load and clean data
print("Loading data...")
df = pd.read_csv("./data/creditcard.csv")
predictions = pd.read_csv("./predictions/predictions.csv")

# Combine original data with predictions and handle NaN values
df['Predicted_Class'] = predictions['Predicted_Class'].fillna(0)
df['Fraud_Probability'] = predictions['Probability_Class_1'].fillna(0)

# Ensure all numeric columns are properly typed
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(0)

# Custom color scheme
colors = {
    'background': '#f8f9fa',
    'text': '#2c3e50',
    'card': '#ffffff',
    'primary': '#3498db',
    'secondary': '#2ecc71',
    'accent': '#e74c3c'
}

# App layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Credit Card Fraud Detection Dashboard", 
                style={'textAlign': 'center', 'color': colors['text'], 'marginBottom': '20px'}),
        html.P("Interactive analysis of fraud detection model performance",
               style={'textAlign': 'center', 'color': colors['text'], 'marginBottom': '30px'})
    ], style={'backgroundColor': colors['card'], 'padding': '20px', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
    
    # Main content
    html.Div([
        # First row - Class distributions
        html.Div([
            html.Div([
                html.H3("Actual vs Predicted Class Distribution",
                       style={'color': colors['text'], 'marginBottom': '15px'}),
                dcc.Graph(id='class-distribution',
                         style={'backgroundColor': colors['card'], 'padding': '10px', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
            ], style={'width': '100%', 'padding': '20px'})
        ], style={'display': 'flex', 'marginBottom': '30px'}),
        
        # Second row - Feature selector and distribution
        html.Div([
            html.Div([
                html.H3("Feature Distribution Analysis",
                       style={'color': colors['text'], 'marginBottom': '15px'}),
                dcc.Dropdown(
                    id='feature-dropdown',
                    options=[{'label': col, 'value': col} for col in df.columns if col.startswith('V') or col == 'Amount'],
                    value='V1',
                    style={'width': '100%', 'marginBottom': '20px', 'backgroundColor': colors['card']}
                ),
                dcc.Graph(id='feature-distribution',
                         style={'backgroundColor': colors['card'], 'padding': '10px', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
            ], style={'width': '100%', 'padding': '20px'})
        ], style={'marginBottom': '30px'}),
        
        # Third row - Probability Distribution
        html.Div([
            html.Div([
                html.H3("Fraud Probability Distribution",
                       style={'color': colors['text'], 'marginBottom': '15px'}),
                dcc.Graph(id='probability-distribution',
                         style={'backgroundColor': colors['card'], 'padding': '10px', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
            ], style={'width': '100%', 'padding': '20px'})
        ], style={'marginBottom': '30px'}),
        
        # Fourth row - Model Performance Metrics
        html.Div([
            html.Div([
                html.H3("Confusion Matrix",
                       style={'color': colors['text'], 'marginBottom': '15px'}),
                dcc.Graph(id='confusion-matrix',
                         style={'backgroundColor': colors['card'], 'padding': '10px', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
            ], style={'width': '50%', 'padding': '20px'}),
            html.Div([
                html.H3("ROC Curve",
                       style={'color': colors['text'], 'marginBottom': '15px'}),
                dcc.Graph(id='roc-curve',
                         style={'backgroundColor': colors['card'], 'padding': '10px', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
            ], style={'width': '50%', 'padding': '20px'})
        ], style={'display': 'flex'})
    ], style={'padding': '20px', 'backgroundColor': colors['background']})
], style={'backgroundColor': colors['background'], 'padding': '20px'})

# Callback for feature distribution
@app.callback(
    Output('feature-distribution', 'figure'),
    [Input('feature-dropdown', 'value')]
)
def update_feature_distribution(feature):
    fig = px.histogram(df, x=feature, color='Class', 
                      title=f'Distribution of {feature} by Class',
                      barmode='overlay',
                      opacity=0.7,
                      color_discrete_sequence=[colors['primary'], colors['accent']])
    fig.update_layout(
        plot_bgcolor=colors['card'],
        paper_bgcolor=colors['card'],
        font_color=colors['text']
    )
    return fig

# Callback for class distribution
@app.callback(
    Output('class-distribution', 'figure'),
    [Input('class-distribution', 'id')]
)
def update_class_distribution(_):
    comparison_data = pd.DataFrame({
        'Category': ['Actual Non-Fraud', 'Actual Fraud', 'Predicted Non-Fraud', 'Predicted Fraud'],
        'Count': [
            len(df[df['Class'] == 0]),
            len(df[df['Class'] == 1]),
            len(df[df['Predicted_Class'] == 0]),
            len(df[df['Predicted_Class'] == 1])
        ],
        'Type': ['Actual', 'Actual', 'Predicted', 'Predicted']
    })
    
    fig = px.bar(comparison_data, x='Category', y='Count', color='Type',
                 title='Actual vs Predicted Class Distribution',
                 barmode='group',
                 color_discrete_sequence=[colors['primary'], colors['secondary']])
    fig.update_layout(
        plot_bgcolor=colors['card'],
        paper_bgcolor=colors['card'],
        font_color=colors['text']
    )
    return fig

# Callback for probability distribution
@app.callback(
    Output('probability-distribution', 'figure'),
    [Input('probability-distribution', 'id')]
)
def update_probability_distribution(_):
    fig = px.histogram(df, x='Fraud_Probability',
                      title='Distribution of Fraud Probabilities',
                      nbins=50,
                      color_discrete_sequence=[colors['primary']])
    fig.update_layout(
        plot_bgcolor=colors['card'],
        paper_bgcolor=colors['card'],
        font_color=colors['text']
    )
    return fig

# Callback for confusion matrix
@app.callback(
    Output('confusion-matrix', 'figure'),
    [Input('confusion-matrix', 'id')]
)
def update_confusion_matrix(_):
    cm = pd.crosstab(df['Class'], df['Predicted_Class'])
    
    fig = px.imshow(cm,
                    labels=dict(x="Predicted", y="Actual"),
                    x=['Non-Fraud (0)', 'Fraud (1)'],
                    y=['Non-Fraud (0)', 'Fraud (1)'],
                    title="Confusion Matrix",
                    color_continuous_scale=['#f8f9fa', colors['primary']])
    
    for i in range(2):
        for j in range(2):
            fig.add_annotation(
                x=j, y=i,
                text=str(cm.iloc[i, j]),
                showarrow=False,
                font=dict(color='white' if cm.iloc[i, j] > cm.values.mean() else colors['text'])
            )
    
    fig.update_layout(
        plot_bgcolor=colors['card'],
        paper_bgcolor=colors['card'],
        font_color=colors['text']
    )
    return fig

# Callback for ROC curve
@app.callback(
    Output('roc-curve', 'figure'),
    [Input('roc-curve', 'id')]
)
def update_roc_curve(_):
    from sklearn.metrics import roc_curve, auc
    
    valid_indices = ~np.isnan(df['Fraud_Probability']) & ~np.isnan(df['Class'])
    y_true = df['Class'][valid_indices]
    y_score = df['Fraud_Probability'][valid_indices]
    
    if len(y_true) > 0 and len(y_score) > 0:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr,
                                name=f'ROC curve (AUC = {roc_auc:.2f})',
                                mode='lines',
                                line=dict(color=colors['primary'], width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                                name='Random',
                                mode='lines',
                                line=dict(dash='dash', color=colors['text'])))
        
        fig.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            plot_bgcolor=colors['card'],
            paper_bgcolor=colors['card'],
            font_color=colors['text']
        )
    else:
        fig = go.Figure()
        fig.add_annotation(
            text="No valid data available for ROC curve",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color=colors['text'])
        )
        fig.update_layout(
            plot_bgcolor=colors['card'],
            paper_bgcolor=colors['card']
        )
    
    return fig

if __name__ == '__main__':
    app.run(debug=True) 