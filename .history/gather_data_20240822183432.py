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

def main():
    initialize_data()
    
chicago_taxi_dataset = ""

def get_data():
    # dataset
    chicago_taxi_dataset = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")

def clean_data():
    # updates dataframe to use specific columns 
    training_data = chicago_taxi_dataset[['TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE']]
    print('Read dataset completed successfully')
    print('total rows: ', len(training_data.index))
    training_data.head(200)

def view_database_statistics():
    # view statistics of the dataset
    print('Total rows: {0}\n\n'.format(len(chicago_taxi_dataset.index)))
    print(chicago_taxi_dataset.describe(include='all'))

def view_correlation_matrix():
    # view correlation matrix
    correlation_matrix = chicago_taxi_dataset.corr(numeric_only=True)
    # Which feature correlates most strongly to the label FARE?
    # ---------------------------------------------------------
    #answer = '''
    #The feature with the strongest correlation to the FARE is TRIP_MILES.
    #As you might expect, TRIP_MILES looks like a good feature to start with to train
    #the model. Also, notice that the feature TRIP_SECONDS has a strong correlation
    #with fare too.
    #'''
    #print(answer)


    # Which feature correlates least strongly to the label FARE?
    # -----------------------------------------------------------
    answer = '''The feature with the weakest correlation to the FARE is TIP_RATE.'''
    print(answer)
    
def question_dataset():
    # questions to ask about the dataset
    max_fare = chicago_taxi_dataset['FARE'].max()
    mean_distance = chicago_taxi_dataset['TRIP_MILES'].mean()
    num_unique_companies = chicago_taxi_dataset['COMPANY'].nunique()
    most_common_payment_type = chicago_taxi_dataset['PAYMENT_TYPE'].value_counts().idxmax()
    missing_values = chicago_taxi_dataset.isnull().sum().sum()
    

def visualize_pair_plot():
    sns.pairplot(training_data, x_vars=["FARE", "TRIP_MILES", "TRIP_SECONDS"], y_vars=["FARE", "TRIP_MILES", "TRIP_SECONDS"])

    

def initialize_data():
    get_data()
    clean_data()
    view_database_statistics()
    question_dataset()