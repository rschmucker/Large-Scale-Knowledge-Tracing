# Assessing the Knowledge State of Online Students

This project provides an efficient analysis framework for the study of knowledge tracing algorithms using various large-scale datasets. It converts the different datasets into a standardized format and provides implementations of various logistic regression and deep learning algorithms. Parts of this codebase were adapted from an earlier [repository]([https://github.com/theophilee/learner-performance-prediction]) by Theophile Gervet. We offer convenient parallel processing capabilities for data preparation and feature extraction. Experiments can be started using the files provided in the scripts folder. This repository is published alongside the paper *Assessing the Performance of Online Students - New Data, New Approaches, Improved Accuracy* ([Link](https://jedm.educationaldatamining.org/index.php/JEDM/article/view/541)).

## Algorithms

The following algorithms are implemented:

* [Item Response Theory (IRT)](https://link.springer.com/book/10.1007/978-1-4757-2691-6)
* [Performance Factors Analysis (PFA)](http://pact.cs.cmu.edu/koedinger/pubs/AIED%202009%20final%20Pavlik%20Cen%20Keodinger%20corrected.pdf)
* [DAS3H](https://arxiv.org/pdf/1905.06873.pdf)
* [Best-LR](https://jedm.educationaldatamining.org/index.php/JEDM/article/download/451/123)
* [Best-LR+](https://arxiv.org/abs/2109.01753)
* [AugmentedLR](https://arxiv.org/abs/2109.01753)
* [DKT](https://arxiv.org/pdf/1506.05908.pdf)
* [SAKT](https://arxiv.org/pdf/1907.06837.pdf)
* [SAINT](https://dl.acm.org/doi/pdf/10.1145/3386527.3405945)  
* [SAINT+](https://arxiv.org/pdf/2010.12042.pdf)

## Datasets

The following datasets are supported:

* EdNet KT3 [Data](https://drive.google.com/file/d/1TVyGIWU1Mn3UCjjeD6bcZ57YspByUV7-/view) / [Meta-Data](https://drive.google.com/file/d/117aYJAWG3GU48suS66NPaB82HwFj6xWS/view)
* [Eedi](https://eedi.com/projects/neurips-education-challenge)
* [Junyi15](https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=1198)
* [Junyi20](https://www.kaggle.com/junyiacademy/learning-activity-public-dataset-by-junyi-academy/tasks)
* Squirrel Ai ElemMath2021

*While this framework can also handle the Junyi20 dataset we want to note that the Junyi20 dataset rounds timestamps to the closest 15 minutes. This prevents an exact reconstruction of the student interaction sequences. Because most of the implemented algorithms are sensitive to the order of student responses the Junyi20 data should be handled with care. The Eedi dataset rounds timestamp information to the closest minute, but upon request the dataset authors provided us with a file allowing for an exact sequence reconstruction.*

## Setup

The easiest way to use this project is to start with a new [Conda](https://docs.conda.io/en/latest/miniconda.html) environment. After that one can install all packages using the provided requirement file. There are two version of the requirement file one for GPU and one for CPU machines.

```
conda create python==3.7.10 -n vedu
conda activate vedu

# use one of the requirement files
pip install -r ./config/requirements_cpu.txt
pip install -r ./config/requirements_gpu.txt

# install PyTorch CPU or GPU version
conda install pytorch=1.8.1 torchvision=0.9.1 torchaudio=0.8.1 cpuonly -c pytorch
conda install pytorch=1.8.1 torchvision=0.9.1 cudatoolkit=10.2 torchaudio=0.8.1 -c pytorch
```

Before starting the experiments download the desired datasets using the links above and put the files into the respecitive directories in the *data* folder. Unzip the compressed data files. 

Important preparation scripts
- *data_preparation*: Converts the raw data into a standardized format
- *feature_preparation*: Extracts various features from the standardized data

The folder *scripts/experiments* contains convenient preparation scripts for each dataset to deal with data and feature preparation. Due to the large size of the datasets one might not want to compute all features at the same time. Most script contain a parameter that specifies the number of available threads. Adapt this parameter to match your own system.

## Training

### Logistic Regression

After data and feature preparation the script *train_lr* can be used to train exisiting logistic regression models and to experiment with combinations of various features to create new models. After training, the resulting model and its performance are stored in the *artifacts* folder. 

### Time-specialiced Logistic Regression

The script *time_specialized* can be used to train multiple specialiced models for different parts of the student interaction sequence. This is a useful technique for mitigating the cold-start problem for new students. The script also allows to evaluate the performance of a "generalist" logistic regression model previously trained using the *train_lr* script during different parts of the learning process.

### Specialiced Logistic Regression for Different Partitions

The script *train_multi_lr.sh* can be used to train multiple specialized for different partitions of the datasets. The individual partitions are induced using the provided feature. The script will first train a separate model for each partition, then combine the predictions made by the individual models and then report overall performance.

### Deep Learning Models

The *scripts/deep* folder contains scripts to train various deep learning models. The implementation of the deep learning models assumes an existing PyTorch installation. The deep learning models can be trained directly from the preprocessed data without requiring the additional feature preparation step. The output of the deep learning experiments is stored in the *results* folder.

## Plotting and Analysis

The *scripts* and the *notebooks* folder both contain code to analyse the output of the trained models. The script *analyse_feature_evaluation* can be used to evaluate the output of the feature evaluation experiments (script in the experiments folder). The notebooks folder contains code to evaluate the performance time-specialized logistic regression models as well as the performance of combinations of multiple models. 

## Citation

If you use this library please cite our paper:

*Schmucker, R., Wang, J., Hu, S., & Mitchell, T. (2022). Assessing the Performance of Online Students - New Data, New Approaches, Improved Accuracy. Journal of Educational Data Mining, 14(1), 1–45.* https://jedm.educationaldatamining.org/index.php/JEDM/article/view/541
