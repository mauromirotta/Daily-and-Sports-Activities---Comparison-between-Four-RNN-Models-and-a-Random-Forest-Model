# Torch library.
import torch

# Metrics score.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_label(index):
  ''' 
    get_label() return the name of the label

    Args:
       index (int): Integer representing the label.

    Returns:
       label(string): The name of the label.
'''
  labels = ['a01', 'a02', 'a03', 'a04', 'a05', 'a06', 'a07', 'a08', 'a09', 'a10',
            'a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19']
  return labels[index]


def compute_metrics(predictions, references):
    ''' 
        compute_metrics() calculates the metrics score.

        Args:
           predictions: The classes predicted by the model.
           references: The real classes.

       Returns:
          Dict: A  dictionary containing the metrics score.
    '''
    acc = accuracy_score(references, predictions)
    precision = precision_score(references, predictions, average='weighted', zero_division=0)
    recall = recall_score(references, predictions, average='weighted', zero_division=0)
    f1 = f1_score(references, predictions, average='weighted', zero_division=0)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

def evaluate(model, dataloader, criterion, device):
    '''
       evaluate() evaluates a model during validation or test.

       Args:
         model: The model to evaluate.
         dataloader: The dataloader containing data.
         criterion: Evaluation parameter.
         device: The device selected for evaluation.

       Returns:
        metrics(dict): A dictionary containing evaluation metric values. 
        predictions: The classes predicted by the model.
        references: The real classes.
    '''
    model.eval()
    running_loss = 0.0
    predictions  = []
    references = [] 
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            pred = torch.argmax(outputs, dim=1)
            predictions.extend(pred.cpu().numpy())
            references.extend(labels.cpu().numpy())
    metrics = compute_metrics(predictions, references)
    metrics['loss'] = running_loss / len(dataloader)
    return metrics, predictions, references

def manage_best_model_and_metrics(model, evaluation_metric, val_metrics, best_val_metric, best_val_metrics, best_model, lower_is_better):
    '''
       manage_best_model_and_metrics() selects the best model and its metrics during the training.

       Args:
         model: The model to check.
         evaluation_metric: The comparison metric.
         val_metrics: Metrics of the model to check.
         best_val_metric: Comparison metric of the previous best model.
         best_val_metrics: All metrics of the previous best model.
         best_model: Previous best model.
         lower_is_better(boolean): Comparison rule. 

       Returns:
         best_val_metric: Comparison metric of the new best model.
         best_model: New best model.
         best_val_metrics: All metrics of the neew best model.        
    '''
    if lower_is_better:
        is_best = val_metrics[evaluation_metric] < best_val_metric
    else:
        is_best = val_metrics[evaluation_metric] > best_val_metric
    if is_best:
        print(f"   New best model found with val {evaluation_metric}: {val_metrics[evaluation_metric]:.4f}")
        best_val_metric = val_metrics[evaluation_metric]
        best_model = model
        best_val_metrics = val_metrics
    return best_val_metric, best_model, best_val_metrics