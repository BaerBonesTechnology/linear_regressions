#@title Train Model using linear regression

# general 
import io

# data
import numpy as np
import pandas as pd

# machine learning
import keras

# data visualization
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns

def make_plots(df, feature_names, lavel_name, model_output, sample_size=200):
    # Create a scatter plot of the actual vs predicted values
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(go.Scatter(x=df[feature_names[0]][:sample_size], y=model_output[:sample_size], mode='markers', name='Actual vs Predicted'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df[feature_names[0]][:sample_size], y=df[lavel_name][:sample_size], mode='markers', name='Actual vs Predicted'), row=1, col=2)
    fig.update_layout(title='Actual vs Predicted', xaxis_title=feature_names[0], yaxis_title=lavel_name)
    fig.show();