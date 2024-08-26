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

# constants
import lib.constants.labels as labels
import lib.constants.graph_type  as gt

def build_model(learning_rate, num_features):
    """Create and compile a simple linear regression model."""
    # Most simple Keras models are sequential.
    model = keras.models.Sequential()

    # Describe the topography of the model
    # the topography of a simple linear regression model 
    # is a single node in a single layer.
    model.add(keras.layers.Dense(units=1, input_shape= (num_features,)))

    # Compile the model topography into code that Keras can efficiently
    # execute. Cpnfigure trainign to minimize the model's mean squared error.
    model.compile(optimizer=keras.optimizers.)