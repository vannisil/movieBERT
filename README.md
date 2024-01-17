# movieBERT - BERT Model for Predicting Tags Movies

In this Repo we use a BERT base Model in order to predict tags movies from a dataset called "mspt_full_data_csv".

## Directory Tree
- **data**: contains a custom dataset loading methods;
    - MovieDataset.py
    - filtered_mt.csv
    - mspt_full_data_csv
- **model**: contains a BERT Model for classifier and metrics for 10 epochs of the Model;
    - BERTClassifier.py
    - metrics_10epochs_2FCL.png
- **preprocessing**: contains a pyhton file for preprocessing of the Dataset;
    - Preprocessing.py
- **training**: contains the training and validation accuracy flow of our Bert Model;
    - Training.py
- **utils**: contains the methods to perform traing, validation and test; 
    - Utils.py
- Inference.py: with this file, you can test the model with the three movie examples written in the "test_text" array.
  N.B.: to test the model, you have to download the fine-tuned BERT from the following Google Drive [link](https://drive.google.com/drive/folders/1NWkrn6-gT-TSUJs-hJcvneqx2Ql7GvIz?usp=sharing) and put the file into the "model" folder;
- movieBERT-Colab.ipynb: the Colab file where we have done all the tests. You can download it, upload on [Google Colab](https://colab.research.google.com) and visualize it.

# Description of the model
Our Bert model aims to predict film tags, given a carefully preprocessed dataset.
The dataset in question is a csv file containing film name, description and genre (maximum 3 types of genres). 
We preprocessed the dataset in order to define tokens to be passed to the model for learning.
We then used a basic Bert model (consisting of 12 Transformer Encoders) and defined suitable hyperparameters and started the training.
We evaluated various variants of the model (3 versions in particular) in order to improve its accuracy.


## Features of our BERT Model 

- Predicting tag movies from a given prompt
- Evaluation of model accuracy
Below, an example of output:
![output](https://i.ibb.co/W6dW8vC/Screenshot-2024-01-10-alle-15-43-40.png)

## Authors

- [Giovanni Silvestri](https://www.github.com/vannisil)
- [Marcello Vangi](https://www.github.com/uzingr)
