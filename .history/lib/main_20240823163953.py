import train_model as trainer
import gather_data as data
import display_data as display

#constants
import constants.features as feature_list
import constants.labels as labels

def main():
    data.initialize_data()

    # Set Hyperparameters
    learning_rate = 0.001
    epochs = 20
    batch_size = 50

    # Specify the feature and the label
    features = feature_list.trip_miles
    lebel - labels.fare_label

    model_1