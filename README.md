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

The pre-processed version of benchmark datasets can be downloaded from [here](https://drive.google.com/dataset.pickle)

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --data_path ./dataset/{data.pickle}
```

## Evaluation

To evaluate our model on {data}, run:

```eval
python eval.py --data_path ./dataset/{data.pickle} --model_path {model_weights.pth} 
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [Our awesome model](https://drive.google.com/model_weight.pth) trained on all five benchmark datasets using default hyperparmeters.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
