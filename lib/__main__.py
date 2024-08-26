from .train_model import *
from .gather_data import *
from .validate_data import *

# constants
from .constants.features import *
from .constants.labels import *


def main():
    run_experiment_1()
    run_experiment_2()
    run_experiment_3()


def run_experiment_1():
    # Set hyper-parameters
    learning_rate = 0.001
    epochs = 20
    batch_size = 50

    # Specify the feature and the label
    features = trip_miles
    chart_label = fare_label

    model = run_experiment(
        global_training_data,
        features,
        chart_label,
        learning_rate,
        epochs,
        batch_size
    )

    predict_values(model, features, chart_label)


def run_experiment_2():
    # Set Hyper-parameters
    learning_rate = 0.001
    epochs = 40
    batch_size = 25

    features = trip_miles
    chart_label = fare_label

    model = run_experiment(
        global_training_data,
        features,
        chart_label,
        learning_rate,
        epochs,
        batch_size
    )

    predict_values(model, features, chart_label)


def run_experiment_3():
    # Set Hyper-parameters
    learning_rate = 0.002
    epochs = 40
    batch_size = 25

    global_training_data['TRIP_MINUTES'] = global_training_data['TRIP_SECONDS'] / 60

    features = miles_and_minutes
    chart_label = fare_label

    model = run_experiment(
        global_training_data,
        features,
        chart_label,
        learning_rate,
        epochs,
        batch_size
    )

    predict_values(model, features, label=chart_label)


def predict_values(model, features, label):
    output = predict_fare(model, global_training_data, features, label)
    show_predictions(output)


if __name__ == "__main__":
    main()
