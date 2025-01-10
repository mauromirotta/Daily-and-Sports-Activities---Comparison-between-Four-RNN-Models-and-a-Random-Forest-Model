# Standard libriaries
import os
import pickle
import shutil
import zipfile

# ML libraries
import numpy as np
import pandas as pd

# Third-party imports
from tqdm import tqdm

# Project libraries
from file_manager import configuration, directory_exists, move_directory

# Set parent path.
parent_path = os.path.dirname(os.getcwd())

def unzip_files(relative_directory):
    '''
       unzip_files() unzips the dataframes from the original dataset directory and generates a directory
       called 'data' with the unzipped files.

       Args:
        relative_directory: Path of the dataset directory. 
    '''
    directory = os.path.join(parent_path, relative_directory)
    extract_to_base = os.path.join(parent_path, 'data')
    os.makedirs(extract_to_base, exist_ok=True)
    for item in tqdm(os.listdir(directory), total=19, desc=' UNZIPPING FILES '):
        if item.endswith('.zip'):
            zip_path = os.path.join(directory, item)
            extract_to = os.path.join(extract_to_base, item.replace('.zip', ''))
            os.makedirs(extract_to, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)

def load_data():
    '''
       load_data() extracts from the 'data' directory a list of all dataframs and a list
       of the corresponding labels.

       Returns:
        df_list (list of pandas dataframes): The list of dataframes.
        label_list (list of int): The list of labels. 
    '''
    base_dir = os.path.join(parent_path, 'data')
    df_list = []
    label_list = []
    for i, activity in tqdm(enumerate(os.listdir(base_dir), start=0), total=19, desc=' LOADING DATA '):
        activity_path = os.path.join(base_dir, activity)
        if os.path.isdir(activity_path):   
            for folder in os.listdir(activity_path):
                folder_path = os.path.join(activity_path, folder)
                if os.path.isdir(folder_path):                     
                    for person in os.listdir(folder_path):
                        person_path = os.path.join(folder_path, person)
                        if os.path.isdir(person_path): 
                            for file in os.listdir(person_path):
                                file_path = os.path.join(person_path, file)
                                if file.endswith('.txt'):
                                   df = pd.read_csv(file_path, sep=None, engine='python', header=None) 
                                   df_list.append(df)
                                   label_list.append(i)
    return df_list, label_list

def generate_statistic_df(df_list):
    '''
       generate_statistic_df() converts the dataframe list in a unique dataframe in order to use it for training 
       non-deep ML models (as Random Forest Classifier).
       Every original dataframe correspond to a single sample of the new dataframe.
       Every column of the single original dataframe correspond to 4 columns of the new dataframe
       that are respectely min value, max value, standard deviation and median of the originary column.

       Args:
        df_list(list of pandas dataframes): List of the original dataframes.

       Returns:
        dataset (pandas dataframe): The statistic single dataframe obtained. 
    '''

    def df_to_sample(df):
        values_list = []
        for column in df.columns:
           values_list.append(df[column].min())
           values_list.append(df[column].max())
           values_list.append(df[column].std())
           values_list.append(df[column].median())
        return pd.DataFrame([values_list])
    
    sample_list = [df_to_sample(df) for df in tqdm(df_list, total=len(df_list), desc=' LOADING STATISTIC DATASET ')]
    dataset = pd.concat(sample_list, axis=0, ignore_index=True)
    return dataset


if __name__ == '__main__':
    '''
        extract_data.py generates serialized dataframes in order to speed up the operation of data processing.
        The goal is to make easier the loading of the raw data avoiding the extraction from the orignal dataset,
        an operation requiring many time that it would be operates everytime.
        File generated:
          - df_list.pkl (list of original raw dataframe);
          - labels.pkl (list of labels);
          - statistic_df.pkl (statistic single dataframe);

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

    print(' START DATA EXTRACTION ')

    # Set the configuration from the configuration file.
    config = configuration(parent_path)

    # Check if the original dataset directory exists.
    source_path = os.path.join(parent_path, config.data.source_dataset_dir)
    directory_exists(source_path, " NO SOURCE DIRECTORY DETECTED --> DOWNLOAD IT FROM : https://github.com/mauromirotta/dataset_dsa.git ")
    
    # ---------------------
    # 2. EXTRACTION
    # ---------------------

     # Unzip dataset files.
    unzip_files(config.data.source_dataset_dir)

    # Load dataframes and labels from the unzipped directory.
    data_loaded, labels = load_data()
    label_column = np.array(labels)

    # Delete no more useful directory
    data_dir_path = os.path.join(parent_path, 'data')
    shutil.rmtree(data_dir_path)

    # Create the single statistic dataframe.
    statistic_df = generate_statistic_df(data_loaded)

    # ---------------------
    # 3. SAVING
    # ---------------------

    # Create directory.
    os.makedirs('data_storage', exist_ok=True)
    os.makedirs('init', exist_ok=True)

    #  Set paths.
    df_list_path = os.path.join('init', 'df_list.pkl')
    labels_path = os.path.join('init', 'labels.pkl')
    statistic_df_path = os.path.join('init', 'statistic_df.pkl')

    # Save results.
    with open(df_list_path, 'wb') as file: pickle.dump(data_loaded, file)
    with open(labels_path, 'wb') as file: pickle.dump(label_column, file)
    with open(statistic_df_path, 'wb') as file: pickle.dump(statistic_df, file)

    # Move init directory in data_storage directory.
    move_directory('init', 'data_storage')

    # Move data_storage directory in the project main directory.
    move_directory('data_storage', parent_path)

    print(' DATA EXTRACTION COMPLETE')