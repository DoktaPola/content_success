<p align="center">
  <img src="https://i.imgur.com/SPYT1zV.png" width="154">
  <h1 align="center">Pre-release content success prediction</h1>
  <p align="center">Pipeline for experiments that implement <b>data preprocessing, augmenting, model training, and model scoring</b> of data
  received from <b>Kinopoisk</b>.
Implemented in Python.</p>
  <p align="center">
	<a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/built%20with-Python3-C45AEC.svg" />
    </a>
    <a href="https://matplotlib.org">
	<img src="https://img.shields.io/badge/bulid with-Matplotlib-7fffd4.svg">
    </a>
    <a href="https://seaborn.pydata.org/">
	<img src="https://img.shields.io/badge/bulid with-Seaborn-F70D1A.svg">
    </a>
    <a href="https://pytorch.org/">
	<img src="https://img.shields.io/badge/bulid with-PyTorch-DFFF00.svg">
    </a>
    <a href="https://scikit-learn.org/">
	<img src="https://img.shields.io/badge/bulid with-Sklearn-FD349C.svg">
    </a>
    <a href="https://numpy.org/doc/stable/index.html">
	<img src="https://img.shields.io/badge/bulid with-NumPy-1589FF.svg">
    </a>
    <a href="https://pandas.pydata.org/">
	<img src="https://img.shields.io/badge/bulid with-Pandas-FFFF00.svg">
    </a>
    <a href="https://scipy.org/">
	<img src="https://img.shields.io/badge/bulid with-SciPy-CCCCFF.svg">
    </a>
  </p>


## Table of contents
- [Project Structure](#structure)
- [Installation](#installation)
- [Usage](#usage)

### **Structure**
* **data/**
    - data.json --> raw_data
* **kp_parsing/**
    - KP_Data_Parsing_latest.ipynb --> skript for Kinopoisk parsing
* **src/**
    * **evaluate/** --> metrics for regression models
    * **models/** --> pre-trained model weights
    * **pipeline/** --> pipelines to unite data processing, model training and predictions and counting scores for it's performance
    * **preprocessing/** --> class for data preprocessing
    - **config.py**
    - **constants.py**
    - **core.py** --> base class transformer
    - **schema.py** --> contains all data columns
    - **train_test_split.py** --> class wrapper of sklearn train_test_split
    - **utils.py** --> additional utils for different classes
* **DL_experiments**
    - **data/covers/** --> images with film/series covers
    - Movies_with_sequels.csv --> data for internationsl movies that have sequels/prequels
    - **models_chp/** --> pre-trained model weights
    - **notebooks/** --> many different experiments with preprocessings and modeling
* **ML_experiments/**
    * **demo_ratings.ipynb** --> demo of usage regression pipeline for *ratings* target
    * **ПЕРЕПИСАТЬ** --> demo of usage
    * **ПЕРЕПИСАТЬ** --> notebook contains almost the entire pipeline code for generating jokes, but without dividing into modules(classes) and preprocessing. It was used in Google Colab to train the model on the GPU.
    * **notebooks/** --> many different experiments with preprocessings and modeling
        * **notebooks_old_targets/**
        * **preproc_modeling/**
        * **preproc/**
        * **TARGET/**

### **Installation**
__Important:__ depending on your system, make sure to use `pip3` and `python3` instead.
**Environment**
* Python version 3.9
* All dependencies are mentioned in *requirements.txt*

### **Usage**
This repository contains 1 pipeline for experiments:
- content success regressor

To run one of pipelines, see the usage **[ratings ml demo](https://github.com/DoktaPola/content_success/blob/main/ML_experiments/demo_ratings.ipynb)**, **[budget ml demo](ADD)**.

* First of all, all needed for pipeline instances have to be called and passed to the appropriate pipeline.
* Then, prepare_data() has to be called, in order to process the dataset before training.
* Next, call train_model().
* And finally, predict().

---

> **Disclaimer**<a name="disclaimer" />: Please Note that this is a research project. I am by no means responsible for any usage of this tool. Use on your own behalf. I'm also not responsible if your accounts get banned due to extensive use of this tool.
