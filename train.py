# Standard libraries
import os
import pickle
import warnings

# Torch library and modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

# Project libraries
from data_classes.dataset import Dataset
from tools.utils import compute_metrics, evaluate, manage_best_model_and_metrics
from tools.file_manager import configuration, file_exists, move_directory
from tools.plots import plot_training_metrics

# Models
from model_classes.stackedrnn_model import StackedRNN
from model_classes.bidirectionalrnn_model import BidirectionalRNN
from model_classes.gru_model import GRU
from model_classes.lstm_model import LSTM
from sklearn.ensemble import RandomForestClassifier

# Manage warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def train_one_epoch(model, train_dl, criterion, optimizer, scheduler, device):
    '''
       train_one_epoch() trains the model for one single epoch.

       Args:
         model: The model to train.
         train_dl: The dataloader of the training set.
         criterion: Training parameter.
         optimizer: Training parameter.
         scheduler: Training parameter.
         device: The device selected for training.

       Returns:
        train_metrics(dict): A dictionary containing train metric values. 
    '''
    model.train()
    running_loss = 0.0
    predictions  = []
    references = [] 
    for inputs, labels in train_dl:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
        pred = torch.argmax(outputs, dim=1)
        predictions.extend(pred.cpu().numpy())
        references.extend(labels.cpu().numpy())
    train_metrics = compute_metrics(predictions, references)
    train_metrics['loss'] = running_loss / len(train_dl)
    return train_metrics

def update_training_record(training_records, train_metrics, val_metrics):
    '''
       update_training_record() merges and update the training and the validation metric results.

       Args: 
       training_records: Previous training metric values that will be update.
       train_metrics: Current epoch training result.
       val_metrics: Current epoch validation result.
         

       Returns:
        training_record(dict): A dictionary containing train metric values. 
    '''
    training_records['train_loss'].append(train_metrics['loss'])
    training_records['train_accuracy'].append(train_metrics['accuracy'])
    training_records['train_precision'].append(train_metrics['precision'])
    training_records['train_recall'].append(train_metrics['recall'])
    training_records['train_f1'].append(train_metrics['f1'])
    training_records['val_loss'].append(val_metrics['loss'])
    training_records['val_accuracy'].append(val_metrics['accuracy'])
    training_records['val_precision'].append(val_metrics['precision'])
    training_records['val_recall'].append(val_metrics['recall'])
    training_records['val_f1'].append(val_metrics['f1'])
    return training_records


def train_model(model, train_dl, val_dl, config, device):
    '''
       train_model() trains and validates a generic deep learning model. Also it sets the training configuration
       parameters.

       Args:
         model: The model to train.
         train_dl: The dataloader of the training set.
         train_dl: The dataloader of the validation set.
         config: Configuration parameters.
         device: The device selected for training.

       Returns:
        best_models(torch model): The best model obtained.
        train_records(dict): A dictionary containing train metrics values for every epochs. 
    '''

    # Set parameters.
    epochs = config.training.epochs
    criterion = nn.CrossEntropyLoss()

    # Set optimizzer.
    if config.training.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config.training.lr)
    elif config.training.optimizer == 'RMSprop':
        optimizer = optim.rmsprop(model.parameters(), lr=config.training.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.training.lr)

    # Set scheduler.
    total_steps = len(train_dl) * epochs
    warmup_steps = int(total_steps * config.training.warmup_ratio)
    scheduler_lambda = lambda step: (step / warmup_steps) if step < warmup_steps else max(0.0, (total_steps - step) / (total_steps - warmup_steps))
    scheduler = LambdaLR(optimizer, lr_lambda=scheduler_lambda)
    
    # Activate model.
    model.to(device)

    # Set comparison method between metrics
    if config.training.best_metric_lower_is_better:
        best_val_metric = float('inf')
    else:
        best_val_metric = float('-inf')

    # Set EarlyStopping parameters.
    early_stopping_count = 0
    early_stopping_metric = config.training.evaluation_metric

    # Variabales where the best model state will be saved.
    best_val_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'loss' :0}
    best_model = None

    # Initialize training records
    training_records = {'train_loss': [], 'train_accuracy': [], 'train_precision': [], 'train_recall': [], 'train_f1': [], 'val_loss': [], 'val_accuracy': [], 'val_precision': [], 'val_recall': [], 'val_f1': []}

    # Training loop
    for epoch in range(epochs):
        print(f"  Epoch {epoch+1}/{epochs}")
        train_metrics = train_one_epoch(model, train_dl, criterion, optimizer, scheduler, device)
        print(f"   Train loss: {train_metrics['loss']:.4f} - Train accuracy: {train_metrics['accuracy']:.4f} - Train precision: {train_metrics['precision']:.4f} - Train recall: {train_metrics['recall']:.4f} - Train f1_score: {train_metrics['f1']:.4f}")
        val_metrics, _, _= evaluate(model, val_dl, criterion, device)
        print(f"   Val loss: {val_metrics['loss']:.4f} - Val accuracy: {val_metrics['accuracy']:.4f} - Val precision: {val_metrics['precision']:.4f} - Val recall: {val_metrics['recall']:.4f} - Val f1_score: {val_metrics['f1']:.4f}")

        # Save training state and best model
        training_records = update_training_record(training_records, train_metrics, val_metrics)
        best_val_metric, best_model, best_val_metrics = manage_best_model_and_metrics(model, config.training.evaluation_metric, val_metrics, best_val_metric, best_val_metrics, best_model, config.training.best_metric_lower_is_better)
        
        # EarlyStopping 
        if early_stopping_metric == 'loss':
            if val_metrics['loss'] > best_val_metrics['loss']:
                early_stopping_count += 1
            else:
                early_stopping_count = 0
        else:
            if val_metrics[early_stopping_metric] < best_val_metrics[early_stopping_metric]:
                early_stopping_count += 1
            else:
                early_stopping_count = 0

        if early_stopping_count >= config.training.early_stopping_epochs:
            print('   Early stopping after ' + str(early_stopping_count) + ' epochs on metric : ' + early_stopping_metric)
            break
    
    print(f"   Best model selected : loss: {best_val_metrics['loss']:.4f} - accuracy: {best_val_metrics['accuracy']:.4f} - precision: {best_val_metrics['precision']:.4f} - recall: {best_val_metrics['recall']:.4f} - f1_score: {best_val_metrics['f1']:.4f}")
    return best_model, training_records

def stacked_model_training(config, device, train_dl, val_dl):
    '''
       stacked_model_training() loads, trains and saves a stacked RNN model.

       Args:
         config: Configuration parameters.
         device: The device selected for training.
         train_dl: The dataloader of the training set.
         train_dl: The dataloader of the validation set.

       Returns:
        training_metrcis(dict): A dictionary containing train metrics values for every epochs. 
    '''

    model = StackedRNN(config.stackedrnn_model.input_layer, 
                            config.stackedrnn_model.hidden_layer,
                            config.stackedrnn_model.output_size,
                            config.stackedrnn_model.num_layer)
    
    print(" STACKED RNN MODEL TRAINING AND VALIDATION")
    best_model, training_metrics = train_model(model, train_dl, val_dl, config, device)
    
    torch.save(best_model.state_dict(), f"{config.training.checkpoint_dir}/stackedrnn_best_model.pt")
    print(" STACKED RNN MODEL SAVED")
    return training_metrics


def bidirectional_model_training(config, device, train_dl, val_dl):
    '''
       bidirectional_model_training() loads, trains and saves a bidirectional RNN model.

       Args:
         config: Configuration parameters.
         device: The device selected for training.
         train_dl: The dataloader of the training set.
         train_dl: The dataloader of the validation set.

       Returns:
        training_metrcis(dict): A dictionary containing train metrics values for every epochs. 
    '''
    model = BidirectionalRNN(config.bidirectionalrnn_model.input_layer, 
                            config.bidirectionalrnn_model.hidden_layer,
                            config.bidirectionalrnn_model.output_size,
                            config.bidirectionalrnn_model.num_layer)
    
    print(" BIDIRECTIONAL RNN MODEL TRAINING AND VALIDATION")
    best_model, training_metrics = train_model(model, train_dl, val_dl, config, device)
    
    torch.save(best_model.state_dict(), f"{config.training.checkpoint_dir}/bidirectionalrnn_best_model.pt")
    print(" BIDIRECTIONAL RNN MODEL SAVED")
    return training_metrics
    

def gru_model_training(config, device, train_dl, val_dl):
    '''
       gru_model_training() loads, trains and saves a GRU RNN model.

       Args:
         config: Configuration parameters.
         device: The device selected for training.
         train_dl: The dataloader of the training set.
         train_dl: The dataloader of the validation set.

       Returns:
        training_metrcis(dict): A dictionary containing train metrics values for every epochs. 
    '''
    model = GRU(config.gru_model.input_layer, 
                 config.gru_model.hidden_layer,
                 config.gru_model.output_size,
                 config.gru_model.num_layer)
   
    print(" GRU MODEL TRAINING AND VALIDATION")
    best_model, training_metrics = train_model(model, train_dl, val_dl, config, device)
    torch.save(best_model.state_dict(), f"{config.training.checkpoint_dir}/gru_best_model.pt")
    print(" GRU MODEL SAVED")
    return training_metrics
    

def lstm_model_training(config, device, train_dl, val_dl):
    '''
       lstm_model_training() loads, trains and saves a LSTM RNN model.

       Args:
         config: Configuration parameters.
         device: The device selected for training.
         train_dl: The dataloader of the training set.
         train_dl: The dataloader of the validation set.

       Returns:
        training_metrcis(dict): A dictionary containing train metrics values for every epochs. 
    '''
    model = LSTM(config.lstm_model.input_layer, 
                 config.lstm_model.hidden_layer,
                 config.lstm_model.output_size,
                 config.lstm_model.num_layer)
    
    print(" LSTM MODEL TRAINING AND VALIDATION")
    best_model, training_metrics = train_model(model, train_dl, val_dl, config, device)
    torch.save(best_model.state_dict(), f"{config.training.checkpoint_dir}/lstm_best_model.pt")
    print(" LSTM MODEL SAVED")
    return training_metrics

def random_forest_model_training(config, X_train, y_train):
    '''
       random_forest_model_training() loads, trains and saves a Random Forest model.

       Args:
         config: Configuration parameters.
         X_train: Data for training.
         y_train: Labels of training data. 
    '''
    model = RandomForestClassifier(n_estimators=config.random_forest.n_estimators,
                                    criterion=config.random_forest.criterion,
                                    max_depth=config.random_forest.max_depth,
                                    min_samples_split=config.random_forest.min_samples_split,
                                    min_samples_leaf=config.random_forest.min_samples_leaf,
                                    max_features=config.random_forest.max_features,
                                    bootstrap=config.random_forest.bootstrap,
                                    oob_score=config.random_forest.oob_score,
                                    n_jobs=config.random_forest.n_jobs,
                                    random_state=config.random_forest.random_state)
    
    print(" RANDOM FOREST MODEL TRAINING ")
    model.fit(X_train, y_train)

    rf_path = os.path.join(config.training.checkpoint_dir,'random_forest_model.pkl')
    with open(rf_path, 'wb') as file: pickle.dump(model, file)
    print(" RANDOM FOREST MODEL SAVED ")
    


if __name__ == '__main__':
    '''
        train.py trains 4 different deep learning models of RNN (Ricurrent Neural Network) and 
        a Random Forest Classification model (non-deep).

        RNN Models:
           - Stacked RNN;
           - Bidirectional RNN;
           - GRU;
           - LSTM;

        Model weights and random forest configurated model will be saved in a checkpoint directory in data_storage.

        train.py also saves in training.plot, within plots directory, training charts considering
        metrics as accuracy, loss, precision, recall and f1.
        
        Summary:
        1. CONFIGURATION
        2. LOAD DATA
        3. TRAIN MODELS
        4. PLOT RESULTS
    '''

    # ---------------------
    # 1. CONFIGURATION
    # ---------------------

    print(" START TRAINING")
    
    # Set the configuration from the configuration file.
    config = configuration()

    # Set model to train list.
    model_to_train = [model.lower() for model in config.model.model_to_train]

    
    # ---------------------
    # 2. LOAD DATA
    # ---------------------

    # Data directory.
    dataset_dir = os.path.join('data_storage', config.data.data_dir)

    # if there is at least one deep model in model to train
    if 'stacked' or 'bidirectional' or 'gru' or 'lstm' in model_to_train:

        # Add Dataset class to torch serializzation safe globals.
        torch.serialization.add_safe_globals({'Safe_Dataset': Dataset})

        # Set file paths.
        train_path = os.path.join(dataset_dir, config.data.train_file_name + '.pkl')
        val_path = os.path.join(dataset_dir, config.data.val_file_name + '.pkl')

        # Check if files exist.
        file_exists(train_path, ' NO TRAINING FILE DETECTED --> TRY TO RUN extract_data.py')
        file_exists(val_path, ' NO VALIDATION FILE DETECTED --> TRY TO RUN extract_data.py')

        # Open files and load elemets.
        with open(train_path, 'rb') as file: train_dataset = pickle.load(file)        
        with open(val_path, 'rb') as file: val_dataset = pickle.load(file)
        
        # Create dataloaders.
        train_dl = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
        val_dl = DataLoader(val_dataset, batch_size=config.training.batch_size,  shuffle=True)
    
    # if random forest is in the list of models to train.    
    if 'random_forest' in model_to_train:

        # Set file paths.
        X_train_path = os.path.join(dataset_dir, config.data.statistic_X_train_file_name + '.pkl')
        y_train_path = os.path.join(dataset_dir, config.data.statistic_y_train_file_name + '.pkl')
        
        # Check if files exist.
        file_exists(X_train_path, ' NO X_TRAIN FILE DETECTED --> TRY TO RUN extract_data.py')
        file_exists(X_train_path, ' NO Y_TRAIN FILE DETECTED --> TRY TO RUN extract_data.py')
        
        # Open files and load elemets.
        with open(X_train_path, 'rb') as file: X_train = pickle.load(file)
        with open(y_train_path, 'rb') as file: y_train = pickle.load(file)  

    
    # ---------------------
    # 3. TRAIN MODELS
    # ---------------------
        
    # Select device.
    if config.training.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(' CURRENT DEVICE : CUDA')
    else:
        device = torch.device('cpu')
        print(' CURRENT DEVICE : CPU')
    
    # Create checkpoints directory (where the results of the training will be saved).
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)

    # Train the chosen models.
    if 'stacked' in model_to_train:
        stacked_model_training_records = stacked_model_training(config, device, train_dl, val_dl)
    if 'bidirectional' in model_to_train:
        bidirectional_model_training_records = bidirectional_model_training(config, device, train_dl, val_dl)
    if 'gru' in model_to_train:
        gru_model_training_records = gru_model_training(config, device, train_dl, val_dl)
    if 'lstm' in model_to_train:
        lstm_model_training_records = lstm_model_training(config, device, train_dl, val_dl)
    if 'random_forest' in model_to_train:
        random_forest_model_training(config, X_train, y_train)

    # Move the checkpoints obtained in a data_storage subdirectory.
    os.makedirs('data_storage', exist_ok=True)
    move_directory(config.training.checkpoint_dir, 'data_storage')

    # ---------------------
    # 4. PLOT RESULTS
    # ---------------------
        
    # Create the directory where the plots will be saved.
    os.makedirs('training_plots', exist_ok=True)

    # Chose metrics to plot.
    metrics_to_plot = config.training.metrics_to_plot

    # Plot the selected models metrics.
    if 'stacked' in model_to_train:
        plot_training_metrics(stacked_model_training_records, metrics_to_plot, config.training.epochs, " STACKED RNN MODEL", 'STACKED_RNN_TRAINING.jpeg')
    if 'bidirectional' in model_to_train:
        plot_training_metrics(bidirectional_model_training_records, metrics_to_plot, config.training.epochs, " BIDIRECTIONAL RNN MODEL", 'BIDIRECTIONAL_RNN_TRAINING.jpeg')
    if 'gru' in model_to_train:
        plot_training_metrics(gru_model_training_records, metrics_to_plot, config.training.epochs, " GRU MODEL", "GRU_MODEL_TRAINING.jpg")
    if 'lstm' in model_to_train:
        plot_training_metrics(lstm_model_training_records, metrics_to_plot, config.training.epochs, " LSTM MODEL", "LSTM_MODEL_TRAINING.jpg")
    
    # Move the plot figure obtained in plots/training_plots directory.
    os.makedirs('plots', exist_ok=True)
    move_directory('training_plots', 'plots')

    # Final print operation.
    print(" TRAINING COMPLETE")