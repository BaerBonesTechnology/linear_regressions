import train_model as trainer
import gather_data as data
import display_data as display

#constants
import constants.features as feature_list
import constants.labels as labels

    data.initialize_data()

    # Set Hyperparameters
    learning_rate = 0.001
    epochs = 20
    batch_size = 50

    # Specify the feature and the label
    features = feature_list.trip_miles
    chart_label = labels.fare_label

    model_1 = trainer.runExperiment(
        data.training_data,
        features,
        chart_label,
        learning_rate,
        epochs,
        batch_size
    )