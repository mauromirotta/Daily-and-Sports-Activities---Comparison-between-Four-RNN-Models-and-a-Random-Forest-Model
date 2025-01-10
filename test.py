# Standard libraries
import os
import pickle
import warnings

# Torch library and modules
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Project library
from data_classes.dataset import Dataset
from tools.plots import plot_metrics, comparison_bar_chart, plot_confusion_matrix
from tools.utils import  compute_metrics, evaluate
from tools.file_manager import configuration, file_exists, move_directory

# Models
from model_classes.stackedrnn_model import StackedRNN
from model_classes.bidirectionalrnn_model import BidirectionalRNN
from model_classes.gru_model import GRU
from model_classes.lstm_model import LSTM

# Manage warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def test_model(model, device, test_dl, title_cm):
    '''
       test_model() tests a deep learning model.

       Args:
         model: The model selected.
         device: The device selected for training.
         test_dl: The dataloader of the test set.
         title_cm: The title of the confusion matrix plot.

       Returns:
        test_metrics(dict): A dictionary containing test metrics values. 
    '''
    criterion = nn.CrossEntropyLoss()
    test_metrics, prediction, references = evaluate(model, test_dl, criterion, device)
    for key, value in test_metrics.items():
        print(f"  Test {key}: {value:.4f}")
    plot_confusion_matrix(prediction, references, title_cm)
    return test_metrics
    
def stacked_model_test(config, device, test_dl):
    '''
       stacked_model_test() loads, tests and saves a stacked RNN model.

       Args:
         config: Configuration parameters.
         device: The device selected for training.
         test_dl: The dataloader of the test set.

       Returns:
        metrics(dict): A dictionary containing test metrics values. 
    '''
    model = StackedRNN(config.stackedrnn_model.input_layer, 
                            config.stackedrnn_model.hidden_layer,
                            config.stackedrnn_model.output_size,
                            config.stackedrnn_model.num_layer)
    model.to(device)


    checkpoint_path = f"{config.training.checkpoint_dir}/stackedrnn_best_model.pt"
    path = os.path.join('data_storage', checkpoint_path)
    file_exists(path, " MISSING STACKED RNN MODEL WEIGHTS " )
    torch.serialization.add_safe_globals({'StackedRnn': StackedRNN})
    model.load_state_dict(torch.load(path))
    print(" STACKED RNN LOADED SUCCESSFULLY")
    metrics = test_model(model, device, test_dl, 'STACKED RNN MODEL')
    return metrics
    

def bidirectional_model_test(config, device, test_dl):
    '''
       bidirectional_model_test() loads, tests and saves a bidirectional RNN model.

       Args:
         config: Configuration parameters.
         device: The device selected for training.
         test_dl: The dataloader of the test set.

       Returns:
        metrics(dict): A dictionary containing test metrics values. 
    '''
    model = BidirectionalRNN(config.bidirectionalrnn_model.input_layer, 
                            config.bidirectionalrnn_model.hidden_layer,
                            config.bidirectionalrnn_model.output_size,
                            config.bidirectionalrnn_model.num_layer)
    model.to(device)

    checkpoint_path = f"{config.training.checkpoint_dir}/bidirectionalrnn_best_model.pt"
    path = os.path.join('data_storage', checkpoint_path)
    file_exists(path, " MISSING BIDIRECTIONAL RNN MODEL WEIGHTS " )
    torch.serialization.add_safe_globals({'BidirectionalRnn': BidirectionalRNN})
    model.load_state_dict(torch.load(path))
    print(" BIDIRECTIONAL RNN LOADED SUCCESSFULLY")
    metrics = test_model(model, device, test_dl, 'BIDIRECTIONAL RNN MODEL')
    return metrics
    


def gru_model_test(config, device, test_dl):
    '''
       gru_model_test() loads, tests and saves a GRU RNN model.

       Args:
         config: Configuration parameters.
         device: The device selected for training.
         test_dl: The dataloader of the test set.

       Returns:
        metrics(dict): A dictionary containing test metrics values. 
    '''
    model = GRU(config.gru_model.input_layer, 
                 config.gru_model.hidden_layer,
                 config.gru_model.output_size,
                 config.gru_model.num_layer)
    model.to(device)
    
    checkpoint_path = f"{config.training.checkpoint_dir}/gru_best_model.pt"
    path = os.path.join('data_storage', checkpoint_path)
    file_exists(path, " MISSING GRU MODEL WEIGHTS " )
    torch.serialization.add_safe_globals({'GRU': GRU})
    model.load_state_dict(torch.load(path))
    print(" GRU LOADED SUCCESSFULLY")
    metrics = test_model(model, device, test_dl, 'GRU MODEL')
    return metrics
    
    

def lstm_model_test(config, device, test_dl):
    '''
       lstm_model_test() loads, tests and saves a LSTM RNN model.

       Args:
         config: Configuration parameters.
         device: The device selected for training.
         test_dl: The dataloader of the test set.

       Returns:
        metrics(dict): A dictionary containing test metrics values. 
    '''
    model = LSTM(config.lstm_model.input_layer, 
                 config.lstm_model.hidden_layer,
                 config.lstm_model.output_size,
                 config.lstm_model.num_layer)
    model.to(device)

    checkpoint_path = f"{config.training.checkpoint_dir}/lstm_best_model.pt"
    path = os.path.join('data_storage', checkpoint_path)
    file_exists(path, " MISSING LSTM MODEL WEIGHTS " )
    torch.serialization.add_safe_globals({'LSTM': LSTM})
    model.load_state_dict(torch.load(path))
    print(" LSTM LOADED SUCCESSFULLY")
    metrics = test_model(model, device, test_dl, 'LSTM MODEL')
    return metrics
    
    
def random_forest_model_test(config, X_test, y_test):
    '''
       random_forest_model_test() loads, tests and saves a Random Forest model.

       Args:
         config: Configuration parameters.
         X_test: Data for test.
         y_test: Labels of test data. 

       Returns:
        metrics(dict): A dictionary containing test metrics values. 
    '''
    checkpoint_path = f"{config.training.checkpoint_dir}/random_forest_model.pkl"
    path = os.path.join('data_storage', checkpoint_path)
    file_exists(path, " MISSING TRAINED RANDOM FOREST MODEL " )
    with open(path, 'rb') as file: model = pickle.load(file)
    print(" RANDOM FOREST MODEL LOADED SUCCESSFULLY")
    predictions = model.predict(X_test)
    metrics = compute_metrics(predictions, y_test)
    for key, value in metrics.items():
        print(f"  Test {key}: {value:.4f}")
    plot_confusion_matrix(predictions=predictions, references=y_test, model_name='RANDOM FOREST MODEL ')
    return metrics



if __name__ == '__main__':
    '''
        test.py tests 4 different deep learning models of RNN (Ricurrent Neural Network) and 
        a Random Forest Classification model (non-deep), evaluating them on 5 different metrics:
           - loss;
           - accuracy;
           - precision;
           - recall;
           - f1 score;
        The plots will be saved in test_plots, in plots directory.
               
        RNN Models:
           - Stacked RNN;
           - Bidirectional RNN;
           - GRU;
           - LSTM;

        Summary:
        1. CONFIGURATION
        2. LOAD DATA
        3. TEST MODELS
        4. FINAL PLOTS
    '''

    # ---------------------
    # 1. CONFIGURATION
    # ---------------------

    print(" START TESTING ")
    
    # Set the configuration from the configuration file.
    config = configuration()

    # # Set model to train list.
    model_to_test = [model.lower() for model in config.model.model_to_test]


    # ---------------------
    # 2. LOAD DATA
    # ---------------------

    # Data directory.
    dataset_dir = os.path.join('data_storage', config.data.data_dir)

    # if there is at least one deep model in model to test
    if 'stacked' or 'bidirectional' or 'gru' or 'lstm' in model_to_test:

        # Add Dataset class to torch serializzation safe globals.
        torch.serialization.add_safe_globals({'Safe_Dataset': Dataset})

        # Set file path.
        test_path = os.path.join(dataset_dir, config.data.test_file_name + '.pkl')

        # Check if file exists.
        file_exists(test_path, ' NO TEST FILE DETECTED --> TRY TO RUN extract_data.py' )

        # Open files and load elemets.
        with open(test_path, 'rb') as file: test_dataset = pickle.load(file)

        # Create dataloader.
        test_dl = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)

    # if random forest is in the list of models to test.    
    if 'random_forest' in model_to_test:

        # Set file paths.
        X_test_path = os.path.join(dataset_dir, config.data.statistic_X_test_file_name + '.pkl')
        y_test_path = os.path.join(dataset_dir, config.data.statistic_y_test_file_name + '.pkl')

        # Check if files exist.
        file_exists(X_test_path, ' NO X_TEST FILE DETECTED --> TRY TO RUN extract_data.py' )
        file_exists(y_test_path, ' NO Y_TEST FILE DETECTED --> TRY TO RUN extract_data.py' )

        # Open files and load elemets.
        with open(X_test_path, 'rb') as file: X_test = pickle.load(file)     
        with open(y_test_path, 'rb') as file: y_test = pickle.load(file)


    # ---------------------
    # 3. TEST MODELS
    # ---------------------
        
    # Select device.
    if config.training.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(' CURRENT DEVICE : CUDA')
    else:
        device = torch.device('cpu')
        print(' CURRENT DEVICE : CPU')
    
    # Create the directory where the plots will be saved.
    os.makedirs('test_plots', exist_ok=True)
    
    # Create a dictionary where the metrics of models will be saved.
    test_model_metrics = {}

    # Test the chosen models.
    if 'stacked' in model_to_test:
        metrics = stacked_model_test(config, device, test_dl)
        plot_metrics(metrics, 'stacked_model_test_metrics.jpeg', " STACKED RNN MODEL TEST METRICS")
        test_model_metrics['Stacked'] = metrics
    if 'bidirectional' in model_to_test:
        metrics = bidirectional_model_test(config, device, test_dl)
        plot_metrics(metrics, 'bidirectional_model_test_metrics.jpeg', " BIDIRECTIONAL RNN MODEL TEST METRICS")
        test_model_metrics['Bidirectional'] = metrics
    if 'gru' in model_to_test:
        metrics = gru_model_test(config, device, test_dl)
        plot_metrics(metrics, 'gru_model_test_metrics.jpeg', " GRU MODEL TEST METRICS")
        test_model_metrics['GRU'] = metrics
    if 'lstm' in model_to_test:
        metrics = lstm_model_test(config, device, test_dl)
        plot_metrics(metrics, 'lstm_model_test_metrics.jpeg', " LSTM MODEL TEST METRICS")
        test_model_metrics['LSTM'] = metrics
    if 'random_forest' in model_to_test:
        metrics = random_forest_model_test(config, X_test, y_test)
        plot_metrics(metrics, 'random_forest_model_test_metrics.jpeg', " RANDOM FOREST MODEL TEST METRICS")
        test_model_metrics['Random Forest'] = metrics


    # ---------------------
    # 4. FINAL PLOTS
    # ---------------------        

    # if there are more than 1 model to test, plot comparison between models.
    if len(model_to_test) > 1:
        comparison_bar_chart(test_model_metrics, 'accuracy')
        comparison_bar_chart(test_model_metrics, 'precision')
        comparison_bar_chart(test_model_metrics, 'recall')
        comparison_bar_chart(test_model_metrics, 'f1')
        comparison_bar_chart(test_model_metrics, 'loss')
    
    # Move the plot figure obtained in plots/test_plots directory.
    os.makedirs('plots', exist_ok=True)
    move_directory('test_plots', 'plots')

    # Final print operation.
    print(" TEST COMPLETE")