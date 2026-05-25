# Adaptive Pseudo-Labeling via Word Coherence for Topic Modeling

This repository is the official implementation of *Adaptive Pseudo-Labeling via Word Coherence for Topic Modeling*. We propose adaptive pseudo-labeling for topic modeling (APT), a self-supervised framework that leverages document embeddings from pretrained transformers, directly reconstructs the Bag-of-Words (BoW) through the inherent relationship between documents and words, dynamically generates pseudo-labels, and integrates proxy-based deep metric learning to enhance coherence and diversity across topics.

<img src="/display/overview_w_background.png" width="672" height="320">

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
|*Wikitext-103* | [Wiki](https://huggingface.co/datasets/Salesforce/wikitext) | 28591 | 10000 | N/A |
|*Web of Science* | [WoS](https://data.mendeley.com/datasets/9rw3vkcfy4/6) | 11967 | 8813 | 7 |
|*Yahoo Answers Topics* | [Yahoo](https://github.com/LC-John/Yahoo-Answers-Topic-Classification-Dataset) | 29156 | 8902 | 10 |

After pre-processing, we divided the dataset into training and testing. Additionally, we removed words that exist only on training or testing datasets. We uploaded the code for dataset pre-processing in a folder named 'dataset'. The pre-processed version of benchmark datasets can be downloaded from [here](https://drive.google.com/drive/folders/1yuqYcuK0KakKaj5wDaDa2eV6y6c_nUCg?usp=sharing)

## Training

To train our APT in the paper, run this command:

```train
python train.py --data_path {data.pickle} --topic 50 --alpha 64. --mrg 0.1
```

## Evaluation

To evaluate our model on {data.pickle}, run:

```eval
python eval.py --data_path {data.pickle} --model_path {model_weights.pth} --tc_topk 15 --td_topk 15 
```

## Pretrained models

You can download pretrained models here:

- [Our model](https://drive.google.com/drive/folders/1juvX9MzwZh9UI0nb9Gna_fUwjuhskTQ0?usp=sharing) trained on all benchmark datasets using default hyperparmeters.

## Usage

### Topic's top-k words

We can extract topic information, specifically the top words for each topic.

```python
model.vocab = {data['vocab_dict']}
model.get_topic_word(top_k=k)

[[topic1_word1, topic1_word2, ... , topic1_wordk],
 [topic2_word1, topic2_word2, ... , topic2_wordk],...]
```

### Embedding visualization

We can visualize embeddings for topics and documents using 'eval.py'. The image is stored in the './output'.

<img src="/output/tSNE.png" width="320" height="320">

## Citation
```bib
@inproceedings{yoon2026apl,
  author = {Yoon, Bohan and Jang, Hyejin},
  title = {Adaptive Pseudo-Labeling via Word Coherence for Topic Modeling},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year = {2026},
  doi = {10.1145/3770855.3817722},
  url = {https://dl.acm.org/doi/10.1145/3770855.3817722}
}
```
