
## DAILY AND SPORTS ACTIVITIES - COMPARISON BETWEEN FOUR RNN MODELS AND A RANDOM FOREST MODEL

This guide aims to assist in understanding the project, which serves as the final exam for the Machine Learning course at Kore University of Enna (Italy). The project is based on the "Daily and Sports Activities" UCI dataset by Billur Barshan and Kerem Altun (link below).

| | |
| --- | --- |
| **Author** | [Mauro Mirotta](https://github.com/mauromirotta) |
| **Organization** | [ Kore University of Enna](https://unikore.it) |
| **License** |  [![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)|
| **Dataset** | [Daily and Sports Activities](https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities) |




---


### Table of Contents

- **Daily and Sports Activities - Comparison between Four RNN Models and a Random Forest Model**
  - Table of Contents
  - Introduction
  - Requirements
  - Getting Started
  - Configuration
  - Code Structure


---

## Introduction

The goal of the project is to train four different types of RNN (Recurent Neural Network) for a classification task, in order to compare their performances to a Random Forest Model.

The data are provided by the "Daily and Sports Activities" dataset (referenced in the section above). The objective of the classification are 19 different types of activities performed by 8 different subjects. The task requires to correctly classify the activity from a time series record of the subject.  

The comparison is based on the following ML metrics: 
- accuracy
- loss
- precision
- recall
- f1 score

The RNN model considered on the project are:
- Stacked RNN
- Bidirectional RNN
- GRU
- LSTM

---

# Requirements

To run the project, you must have the following tools:

- **Python**: The project is based on Python 3.11.2. [**Download the latest version**](https://www.python.org/downloads/). 

- **Git**: The project requires Git for cloning the dataset repository during the preparation. [**Download Git**](https://git-scm.com/).

- **Git Bash**: To run the project scripts, Git Bash is suggested (especially for Windows users). The [download link](https://git-scm.com/) is the same of Git. 

In addiction, the project autonomously installs the python library specified on *requirements.txt*.

---

















## Getting Started

First step after cloning the project repository is to run:

```bash
  bash prepare.sh
```

This command installs the required python libraries, clone the dataset repository and run  *tools/extract_data.py* which extracts the raw data from the dataset repository and converts them into pkl file in order to facilitate the management by the other script. The pkl file will be saved in *data_storage/init* directory.

> [!IMPORTANT]  
> Before getting started with the following scripts it is important to set the parameters on configuration file. Find the explanation on the **Configuration** paragraph below .
---

The following steps represent the main issues of the project. They are:
- **preprocessing.py**
- **train.py**
- **test.py**

All the scripts generates plots in the plots directory. 

The first script you need to run is **preprocessing.py**: 

```bash
  python preprocessing.py
```

That processes data and generates the train, validation and test set and the preprocessing plots.

After doing the preprocessing, you can get started with the training of the model.

run **train.py** script to train the models you have selected in the configuration file.

```bash
  python train.py
```

while for testing the models, run **test.py**:

```bash
  python test.py
```

All scripts generate plot charts on plot directory.

---

## Configuration

To configurate the training and model parameters you can use the configuration file. The file is named *base_config.yaml* and it is located in *config* directory. 

---

> [!IMPORTANT]  
> You do not need to pass the configuration file as a parameter when you run the scripts. The python script access to the file automatically.
---

> [!WARNING]  
> Do not change the interior structure of the configuration file. Just modify the fields respecting the values format. 
---

## Code Structure

The code is structured as follows:

```
main_repository/
|
├── config/
|   ├── base_config.yaml
|
|
├── data_classes/
|   ├── dataset.py
|
|
├── model_classes/
|   ├── bidirectionalrrn_model.py
|   ├── gru_model.py
|   ├── lstm_model.py
|   ├── stackedrnn_model.py
|
|
├── tools/
|   ├── extract_data.py
|   ├── file_manager.py
|   ├── plots.py
|   ├── utilss.py
|
|
├── LICENCE
├── .gitignore
├── prepare.sh
├── README.md
├── requirememts.txt
|
├── preprocessing.py
├── train.py
├── test.py
```

> [!IMPORTANT]  
> After the execution of the scripts new folders will be generated

```

├── data_storage/
|   ├── checkpoints/
|   ├── dataset_dir/
|   ├── init/
|
|
├── dataset_dsa/
|
|
├── plots/
|   ├── preprocessing_plots/
|   ├── train_plots/
|   ├── test_plots/
```














