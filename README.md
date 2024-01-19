<p align="center">
    <img src="https://i.ibb.co/vPrn2ff/OIG-Ph6w-Q7-WBAp-LZa8-UZJpr-W.jpg" width="200px" heigth="200px"/>
</p>


# movieBERT - BERT Model for Predicting Tags Movies

In our project we use the BERT base pre-trained model in order to predict tags for movies from a dataset called "mspt_full_data_csv".

## Directory Tree
- **data**: contains a custom dataset loading methods;
    - MovieDataset.py
    - filtered_mt.csv
    - mspt_full_data_csv
- **model**: contains the BERT model class for the classifier and the best metrics for 10 epochs of training;
    - BERTClassifier.py
    - best_epoch.png
- **preprocessing**: contains a pyhton file used to preprocess the dataset;
    - Preprocessing.py
- **training**: contains the training and validation flow of our BERT model;
    - Training.py
- **utils**: contains the methods to perform traing, validation and test; 
    - Utils.py
- _Inference.py_: with this file, you can test the model with the three movie examples written in the "test_text" array;
- _movieBERT-Colab.ipynb_: the Colab file where we have done all the tests. You can upload it on [Google Colab](https://colab.research.google.com) and visualize all the outputs. You can also check the Colab file at this [link](https://colab.research.google.com/drive/1Mr68cP71SS5rYXcKaNydhZ1sJuF_n0Cb?usp=sharing)

N.B.: to test the model, you have to download the fine-tuned BERT .pth file from the following Google Drive [link](https://drive.google.com/drive/folders/1NWkrn6-gT-TSUJs-hJcvneqx2Ql7GvIz?usp=sharing) and put the file into the "model" folder.

# Description of the model
Our Bert model aims to predict films tags, given a carefully preprocessed dataset.
The dataset in question is a csv file containing film name, description and genre (we have chosen maximum 5 types of genres). 
We preprocessed the dataset in order to define tokens to be passed to the model for learning.
We then used the BERT base model (consisting of 12 Transformer Encoders) and defined suitable hyperparameters and started the training.
We evaluated various variants of the model (3 versions in particular) in order to improve its accuracy. We selected one of them.


## Features of our BERT Model 

- Predicting tag movies from a given prompt.
Below, an example of its output:
![output](https://i.ibb.co/W6dW8vC/Screenshot-2024-01-10-alle-15-43-40.png)

## Canva presentation

You can see the project presentation at this [link](https://www.canva.com/design/DAF52mLNHP4/m6oRdloxEGFHSME1BU-7gQ/edit?utm_content=DAF52mLNHP4&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

## Authors

- [Giovanni Silvestri](https://www.github.com/vannisil)
- [Marcello Vangi](https://www.github.com/uzingr)
