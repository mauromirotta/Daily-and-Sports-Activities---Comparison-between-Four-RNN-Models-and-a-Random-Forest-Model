# Standard libraries
import os
import pickle

# ML libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Torch library and modules
import torch
from torch.utils.data import random_split

# Project libraries
from data_classes.dataset import Dataset
from tools.plots import class_distribution_pie_chart, features_plot,  box_plot, pca_plot
from tools.file_manager import directory_exists, file_exists, configuration, move_directory


def check_NaN_values(df_list):
    """
       check_NaN_values() check if there are sequences that contains NaN values.

       Args:
        df_list (list of pandas dataframes): The list of dataframes.

        Returns:
        True: There is no NaN values.
        False: Some sequences contain NaN values.
    """
    flag = True
    for i, df in enumerate(df_list):
        if df.isna().any().any():
            flag = False
            print(f"DataFrame {i} contains NaN values.")
    return flag 

def check_sequences_length(df_list):
    """
       check_sequences_length() check if all sequences have the same length.

       Args:
        df_list (list of pandas dataframes): The list of dataframes.

        Returns:
        True: All the squences have the same length.
        False: All the sequences do not have the same length.
    """
    num_rows = [df.shape[0] for df in df_list]  
    all_same_length = all(rows == num_rows[0] for rows in num_rows)
    return all_same_length
    
def normalize(data, statistic_data):
    """
       normalize() uses StandardScaler (scikit-learn) to normalize the data distribution.

       Args:
        data (list of pandas dataframes): The list of sequences.
        statistic_data (pandas dataframe): Single dataframe created from the list of dataframes using statistic methods.

        Returns:
        normalize_data: List of sequences normalized.
        statistic_data_scaled: Single statistic dataframes normalized
    """
    df_to_fit =  pd.concat(data, ignore_index=True)
    scaler = StandardScaler()
    scaler.fit(df_to_fit)
    normalized_data = [scaler.transform(df) for df in data]
    statistic_data_scaled = scaler.fit_transform(statistic_data)
    return normalized_data, statistic_data_scaled

if __name__ == '__main__':
    '''
        preprocessing.py includes preprocessing, normalization, torch conversion and data splitting operations.
        The code creates 7 pkl files containing the processed data that will be used during 
        the training and test of the deep ML models and non-deep model.
        pkl files:
          - training.pkl
          - validation.pkl
          - test.pkl
          - statistic_X_train.pkl
          - statistic_y_train.pkl
          - statistic_X_test.pkl
          - statistic_y_test.pkl
        pkl filese will be saved in dataset_dir, in data_storage directory.
        The code also generates 3 charts refered to the data:
          - Classes distribution pie chart;
          - Box plot;
          - PCA plot;
        Charts will be saved on preprocessing_plots directory within plots directory.

        Summary:
        1. CONFIGURATION
        2. LOAD RAW DATA
        3. PREPROCESSING
        4. DATA SPLIT
        5. SAVING
    '''

    # ---------------------
    # 1. CONFIGURATION
    # ---------------------

    print(' START PREPROCESSING ')

    # Check if the  initialization directory exists.
    directory_exists('data_storage\init', " NO INITIALIZATION DIRECTORY DETECTED --> RUN extract_data.py ")
    
    # Set the configuration from the configuration file.
    config = configuration()

    # Create the directory where the plots will be saved.
    os.makedirs('preprocessing_plots', exist_ok=True)

    # Create the direcotry where the datasets will be saved.
    dataset_dir = config.data.data_dir
    os.makedirs(dataset_dir, exist_ok=True)

    # ---------------------
    # 2. LOAD RAW DATA
    # ---------------------

    # Set file paths.
    df_list_path = 'data_storage\init\df_list.pkl'
    labels_path = 'data_storage\init\labels.pkl'
    statistic_df_path = 'data_storage\init\statistic_df.pkl'

    # Check if files exist.
    file_exists(df_list_path, ' NO INITIALIZATION FILE DETECTED --> TRY TO RUN extract_data.py')
    file_exists(labels_path, ' NO INITIALIZATION FILE DETECTED --> TRY TO RUN extract_data.py')
    file_exists(statistic_df_path, ' NO INITIALIZATION FILE DETECTED --> TRY TO RUN extract_data.py')

    # Open files and load elemets.
    with open(df_list_path, 'rb') as file: df_list = pickle.load(file)        
    with open(labels_path, 'rb') as file: labels = pickle.load(file)
    with open(statistic_df_path, 'rb') as file: statistic_df = pickle.load(file)

    # ---------------------
    # 3. PREPROCESSING
    # ---------------------

    # Check if dataframes contain NaN values.
    if check_NaN_values(df_list):
        print(' NO NaN VALUES DETECTED ')

    # Check if all sequences have the same length, considering a dataframe as a sequence.
    if check_sequences_length(df_list):
        print(' ALL SEQUENCES HAVE THE SAME LENGTH')

    print(' START GENERATING PLOTS')

    # Create and save plots of class distribution on 'preprocessing_plots' directory.
    class_distribution_pie_chart(labels)

    # Check the outlier values

    # Plot casual dataframes
    features_plot(df_list)

    # Check the out-layer values by box plot.
    box_plot(df_list)

    # Normalize the dataframes using Z-score normalization
    data_scaled, sta_data_scaled = normalize(df_list, statistic_df)

    # PCA components plot    
    pca_plot(sta_data_scaled, labels)


    # ---------------------
    # 4. DATA SPLIT
    # ---------------------

    # Create torch tensors of scaled data and labels and the Dataset class instance 
    # to represent the dataset
    data_tensor = torch.tensor(np.array(data_scaled), dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    dataset = Dataset(data_tensor, label_tensor)

    # Split of the deep models data
    train_size = int(config.data.train_ratio * len(dataset))
    val_size = int(config.data.val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    # Split of the non-deep models data using train_test_split
    X_train, X_test, y_train, y_test = train_test_split(sta_data_scaled, labels,
                                         train_size=(config.data.train_ratio + config.data.val_ratio), 
                                         random_state=42)
    

    # ---------------------
    # 5. SAVING
    # ---------------------

    # Save deep models datasets.
    train_path = os.path.join(dataset_dir, config.data.train_file_name + '.pkl')
    with open(train_path, 'wb') as file: pickle.dump(train_dataset, file)
    val_path = os.path.join(dataset_dir, config.data.val_file_name + '.pkl')
    with open(val_path, 'wb') as file: pickle.dump(val_dataset, file)
    test_path = os.path.join(dataset_dir, config.data.test_file_name + '.pkl')
    with open(test_path, 'wb') as file: pickle.dump(test_dataset, file)
    
    # Save non-deep model data.
    X_train_path = os.path.join(dataset_dir, config.data.statistic_X_train_file_name + '.pkl')
    with open(X_train_path, 'wb') as file: pickle.dump(X_train, file)
    X_test_path = os.path.join(dataset_dir, config.data.statistic_X_test_file_name + '.pkl')
    with open(X_test_path, 'wb') as file: pickle.dump(X_test, file)
    y_train_path = os.path.join(dataset_dir, config.data.statistic_y_train_file_name + '.pkl')
    with open(y_train_path, 'wb') as file: pickle.dump(y_train, file)
    y_test_path = os.path.join(dataset_dir, config.data.statistic_y_test_file_name + '.pkl')
    with open(y_test_path, 'wb') as file: pickle.dump(y_test, file)
    
    # Move the plot figure obtained in plots/preprocessing_plots directory
    os.makedirs('plots', exist_ok=True)
    move_directory('preprocessing_plots', 'plots')

    # Move the created datasets directory to the data_storage directory.
    move_directory(dataset_dir, 'data_storage')

    # Final print operation
    print(" PREPROCESSING COMPLETE")