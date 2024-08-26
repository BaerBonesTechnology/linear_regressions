# general 
import io

import keras
# data
import pandas as pd
# data visualization
from .display_data import *

# constants
from .constants.model_variables import *


def build_model(learning_rate, num_of_features):
    """Create and compile a simple linear regression model."""
    # Most simple Keras models are sequential.
    model = keras.models.Sequential()

    # Describe the topography of the model
    # the topography of a simple linear regression model 
    # is a single node in a single layer.
    model.add(keras.layers.Dense(units=1, input_shape=(num_of_features,)))

    # Compile the model topography into code that Keras can efficiently
    # execute. Configure training to minimize the model's mean squared error.
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
                  loss=MSE,
                  metrics=[keras.metrics.RootMeanSquaredError()]
                  )

    return model


def train_model(model, data, features, label, epochs, batch_size):
    """Train the model by feeding it data."""

    # Feed the model the feature and label
    # the model will train for the specified number of epochs.
    # input_x = data.iloc[:,1:3].values
    # data[feature]
    history = model.fit(
        x=features,
        y=label,
        batch_size=batch_size,
        epochs=epochs
    )

    # Gather the trained models weight and bias.
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]

    # store the list of epochs separately
    epochs = history.epoch

    # Isolate the error for each epoch
    hist = pd.DataFrame(history.history)

    # To track the progression of training, we're going to take a snapshot of the models root
    # mean squared error at each epoch
    print(hist)
    rmse = hist[RMSE]

    return trained_weight, trained_bias, epochs, rmse


def run_experiment(data, feature_names, label_name, learning_rate, epochs, batch_size):
    print('Info: Starting training experiment with features={} and label={}\n'.format(feature_names,
                                                                                      label_name))

    num_of_features = len(feature_names)

    features = data.loc[:, feature_names].values
    label = data[label_name].values

    model = build_model(learning_rate=learning_rate, num_of_features=num_of_features)
    model_output = train_model(model=model, data=data, features=features, label=label,
                               epochs=epochs, batch_size=batch_size)
    print('\nSUCCESS: training experiment complete\n')
    print('{}'.format(data, feature_names, label_name, model_output))
    make_plots(data, feature_names, label_name, model_output)

    return model