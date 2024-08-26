#@title Train Model using linear regression

# general 
import io

# data
import numpy as np
import pandas as pd
import gather_data as gd

# machine learning
import keras

# data visualization
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns

def make_plots(df, feature_names, label_name, model_output, sample_size=200):

    # get random sample of data
    random_sample = df.sample(n=sample_size).copy()
    random_sample.reset_index()
    weights, bias, epochs, rmse = model_output

    # create figure
    is_2d_plot = len(feature_names) == 1
    model_plot_type = "scatter" if is_2d_plot else "surface"
    fig = make_subplots(rows = 1, cols = 2, subplot_titles=("Loss Curve", "Model Plot"),
                        specs=[[{"type": "scatter"}, {"type": model_plot_type}]])
    plot_data(random_sample, feature_names, label_name, fig)
    plot_model(random_sample, feature_names, weights, bias, fig)
    plot_loss_curve(epochs, rmse, fig)

    fig.show()
    return

def plot_loss_curve(epochs, rmse, fig):
    curve = px.line(x=epochs, y=rmse)
    curve.update_traces(line_color='#636EFA', line_width=3)

    fig.append_trace(curve.data[0], row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_yaxes(title_text="Root Mean Squared Error", row=1, col=1)

    return

def plot_data(df, feature_names, label_name, fig):
    if len(features) ==1:
        scatter = px.scatter(df, x = feature_names[0], y=label_name)
    else:
        scatter = px.scatter_3d(df, x=feature_names[0], y=feature_names[1], z=label_name)

