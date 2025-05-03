# Adaptive Pseudo-Labeling via Word Coherence for Topic Modeling

This repository is the official implementation of *Adaptive Pseudo-Labeling via Word Coherence for Topic Modeling*. We propose adaptive pseudo-labeling for topic modeling (APT), a self-supervised framework that incorporates deep metric learning to improve topic quality.

<img src="/display/framework.png" width="672" height="320">

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Get started

The following lists the statistics of the datasets we used.

| Dataset | Source link | Docs | Words | Categories |
| :----- | :-----: | :-----: | :-----: | :-----: |
|*20Newsgroups* | [20NG](http://qwone.com/~jason/20Newsgroups/) | 18846 | 9994 | 20 |
|*AG's News* | [AG](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html) | 38280 | 8727 | 4 |
|*Wikitext-103* | [Wiki](https://developer.ibm.com/exchanges/data/all/wikitext-103/) | 28591 | 10000 | N/A |
|*Web of Science* | [WoS](https://data.mendeley.com/datasets/9rw3vkcfy4/6) | 11967 | 8813 | 7 |
|*Yahoo Answers Topics* | [Yahoo](https://github.com/LC-John/Yahoo-Answers-Topic-Classification-Dataset) | 29156 | 8902 | 10 |

After pre-processing, we divided the dataset into training and testing. Additionally, we removed words that exist only on training or testing datasets. We uploaded the code for dataset pre-processing in a folder named 'dataset'. The pre-processed version of benchmark datasets can be downloaded from [here](https://drive.google.com/dataset.pickle)

## Training

To train our APT in the paper, run this command:

```train
python train.py --data_path {data.pickle}
```

## Evaluation

To evaluate our model on {data.pickle}, run:

```eval
python eval.py --data_path {data.pickle} --model_path {model_weights.pth} 
```

## Pre-trained Models

You can download pretrained models here:

- [Our model](https://drive.google.com/model_weight.pth) trained on all five benchmark datasets using default hyperparmeters.

## Usage

We can extract the top words and their corresponding probabilities for each topic.

```eval
python eval.py --data_path {data.pickle} --model_path {model_weights.pth} 
```

Our model achieves the following performance on :

<img src="/output/tSNE.png" width="320" height="320">

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 

## Citation
```bib
@inproceedings{}
```
