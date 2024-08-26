import train_model as trainer
import gather_data as data
import display_data as display

def main():
    data.initialize_data()

    # Set Hyperparameters
    learning_rate = 0.001
    epochs = 20
    batch_size = 50

    # 
    features