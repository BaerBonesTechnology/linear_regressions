from train_model import *
from gather_data import *
# constants
from .constants.features import *
from .constants.labels import *

initialize_data()

# Set hyper-parameters
learning_rate = 0.001
epochs = 20
batch_size = 50

# Specify the feature and the label
features = trip_miles
chart_label = fare_label

model_1 = run_experiment(
    global_training_data,
    features,
    chart_label,
    learning_rate,
    epochs,
    batch_size
)
