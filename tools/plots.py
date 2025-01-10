# Standard library
import os

# ML libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

# Graphic third-party libraries
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# Project library
from tools.utils import get_label 

def class_distribution_pie_chart(labels):
    '''
       class_distirbution_pie_chart() generates a pie chart of the distribution of the classes.

         Args:
         labels: Label list.
    '''
    cmap = cm.get_cmap('tab20c')
    colors = [cmap(i) for i in np.linspace(0, 1, 19)]
    classes = [get_label(i) for i in range(0, 19)]
    instance_dict = {i: 0 for i in range(0, 19)}
    for label in labels:  instance_dict[label] += 1
    instances = [instance_dict[i] for i in range (0, 19)]
    plt.figure(figsize=(8, 8))
    plt.pie(instances, labels=classes, autopct='%1.2f%%', shadow=True, textprops={'color': 'black'}, colors=colors)
    plt.title('INSTANCES FOR CLASSES')
    path = os.path.join('preprocessing_plots', 'CLASSES DISTRIBUTION PIE CHART.jpeg')
    plt.savefig(path, format='jpeg')

def features_plot(df_list):
  '''
       features_plots() generates plots of 6 different dataframes of 6 different activities.

         Args:
         df_list: List of all  dataframes.
    '''
  df_to_plot = [df_list[0], df_list[488], df_list[5290], df_list[6369], df_list[8640], df_list[9000]]
  titles = ['SITTING', 'STANDING','WALKING ON A TREADMILL', 'EXERCISING ON A CROSS TRAINER', 'ROWING', 'PLAYING BASKETBALL']
  cmap = cm.get_cmap('gist_ncar', 45)
  colors = cmap(np.linspace(0, 1, 45))
  _,axs = plt.subplots(2, 3, figsize=(18, 10))
  axs = axs.flatten()
  for i in range(6):
    for j, column in enumerate(df_to_plot[i].columns):
       axs[i].plot(df_to_plot[i].index, df_to_plot[i][column], color=colors[j])
       axs[i].set_title(titles[i])
  plt.tight_layout(rect=[0, 0, 1, 0.95])
  plt.suptitle('RANDOM DF FEATURES PLOT')
  path = os.path.join('preprocessing_plots', 'RANDOM DF FEATURES PLOT.jpeg')
  plt.savefig(path, format='jpeg')


def box_plot(df_list):
  '''
       features_plots() generates a box plot of all features.

         Args:
         df_list: List of all  dataframes.
    '''
  df = pd.concat(df_list, axis=0, ignore_index=True)
  plt.figure(figsize=(15, 10))
  sns.boxplot(data=df)
  plt.title('DATA BOX PLOT')
  path = os.path.join('preprocessing_plots', 'DATA BOX PLOT.jpeg')
  plt.savefig(path, format='jpeg')
  
def pca_plot(df, labels):
    '''
       pca_plot() generates plots of 6 different dataframes of 6 different activities.

         Args:
         df: The statistic unique dataset.
         labels: Class labels.
    '''
    pca = PCA(n_components=2)
    df_components = pca.fit_transform(df)
    plt.figure(figsize=(13, 10))
    unique_classes = np.unique(labels)
    colors = plt.cm.get_cmap('tab20c', len(unique_classes))
    for i, class_value in enumerate(unique_classes):
        plt.scatter(df_components[labels == class_value, 0], df_components[labels == class_value, 1], 
                    label=f'Class {get_label(class_value)}', color=colors(i))
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('PCA PLOT')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Classes')
    plt.tight_layout()
    path = os.path.join('preprocessing_plots', 'PCA PLOT.jpeg')
    plt.savefig(path, format='jpeg')

def plot_training_metrics(records, metrics, epochs, name, file_name):
  '''
    plot_training_metrics() plots the learning curve of the selected metrics durign the training epochs.

      Args:
      record: Values of the metrics during the epochs.
      metrics: Metrics to plot.
      epochs: Training epochs.
      name: The name of the chart.
      file_name: The name which the file will be saved.        
  '''
  title = name + " TRAINING AND VALIDATION METRICS "
  fig = go.Figure()
  if 'loss' in metrics:
    fig.add_trace(go.Scatter(x=list(range(1, epochs)), y=records['train_loss'], line=dict(color='red', width=2), name='Train Loss'))
    fig.add_trace(go.Scatter(x=list(range(1, epochs)), y=records['val_loss'], line=dict(color='black', width=2, dash='dash'), name='Val Loss'))
  if 'accuracy' in metrics:
    fig.add_trace(go.Scatter(x=list(range(1, epochs)), y=records['train_accuracy'], line=dict(color='green', width=2), name='Train Accuracy'))
    fig.add_trace(go.Scatter(x=list(range(1, epochs)), y=records['val_accuracy'], line=dict(color='magenta', width=2, dash='dash'), name='Val Accuracy'))
  if 'precision' in metrics:
    fig.add_trace(go.Scatter(x=list(range(1, epochs)), y=records['train_precision'], line=dict(color='orange', width=2), name='Train Precision'))
    fig.add_trace(go.Scatter(x=list(range(1, epochs)), y=records['val_precision'], line=dict(color='yellow', width=2, dash='dash'), name='Val Precision'))
  if 'recall' in metrics:
    fig.add_trace(go.Scatter(x=list(range(1, epochs)), y=records['train_recall'], line=dict(color='blue', width=2), name='Train Recall'))
    fig.add_trace(go.Scatter(x=list(range(1, epochs)), y=records['val_recall'], line=dict(color='gray', width=2, dash='dash'), name='Val Recall'))
  if 'f1' in metrics:
    fig.add_trace(go.Scatter(x=list(range(1, epochs)), y=records['train_f1'], line=dict(color='violet', width=2), name='Train F1'))
    fig.add_trace(go.Scatter(x=list(range(1, epochs)), y=records['val_f1'], line=dict(color='cyan', width=2, dash='dash'), name='Val F1'))
  fig.update_layout(title_text=title, font_size=15, xaxis_title='EPOCHS')
  path = os.path.join('training_plots', file_name)
  fig.write_image(path)

def plot_metrics(metrics, file_name, title):
    '''
    plot_metrics() generates a bar chart of a model metrics.

        Args:
          metrics: Metrics to plot.
          file_name: The name which the file will be saved.
          title. The title of the plot.
    '''
    names = list(metrics.keys())
    values = list(metrics.values())
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, len(names))]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values, color=colors)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.4f}", ha='center', va='bottom')
    plt.xlabel('METRICS')
    plt.ylabel('VALUES')
    plt.title(title)
    plt.tight_layout()
    path = os.path.join('test_plots', file_name)
    plt.savefig(path, format='jpeg')

def comparison_bar_chart(models, metric):
  '''
    comparison_bar_chart() generates a bar chart to compare different model on a single metric .

        Args:
          models:
          metric: The comparison metric.
  '''
  if metric == 'loss':
    del models['Random Forest']
  plot_dict = {}
  for model, metrics in models.items():
    plot_dict[model] = metrics[metric]
  title = " MODELS' " + metric.upper() + " COMPARISON"
  file_name = title.lower() + ".jpeg"
  plot_metrics(plot_dict, file_name, title)

def plot_confusion_matrix(predictions, references, model_name):
  '''
    plot_confusion_metric() generates a confusion metric plot of a model.

    Args:
      predictions: The classes predicted by the model.
      references: The real classes.
      model_name: The model name.
  '''
  cm = confusion_matrix(references, predictions)
  labels = [get_label(i) for i in range(0, 19)]       
  plt.figure(figsize=(13, 11))
  sns.heatmap(cm, annot=True, fmt='d', cmap="Greens", xticklabels=labels, yticklabels=labels)
  title = model_name.upper() + " CONFUSION MATRIX"
  plt.title(title)
  plt.xlabel(' PREDICTIONS')
  plt.ylabel(' REFERENCES')
  plt.xticks(rotation=45, ha="right")
  file_name = title.lower() + ".jpeg"
  path = os.path.join('test_plots', file_name)
  plt.savefig(path, format='jpeg')