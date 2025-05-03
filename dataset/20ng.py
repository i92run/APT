from sklearn.datasets import fetch_20newsgroups
import numpy as np
import pandas as pd
from tqdm import tqdm
import nltk
from sentence_transformers import SentenceTransformer
import pickle
import argparse
from topmost import Preprocess


parser = argparse.ArgumentParser(description='dataset preprocessing')
parser.add_argument('--dataset', type=str, default='20ng', help='dataset name')
parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2', help='encode model')
parser.add_argument('--vocab_size', type=int, default=10000, help='topmost vocab size')
args = parser.parse_args()


def main():
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')

    # load 20ng data
    data = fetch_20newsgroups(subset='all')
    train_data = fetch_20newsgroups(subset='train')
    test_data = fetch_20newsgroups(subset='test')

    # convert data to dataframe
    train_df = pd.DataFrame({'news': train_data.data, 'target': train_data.target})
    test_df = pd.DataFrame({'news': test_data.data, 'target': test_data.target})
    train_df['train'] = True
    test_df['train'] = False
    df = pd.concat([train_df, test_df])
    df['target_name'] = df['target'].apply(lambda x: data.target_names[x])

    # create tqdm for pandas
    tqdm.pandas()

    # generate document embeddings using pretrained transformer
    model = SentenceTransformer(args.model)
    df['emb'] = df['news'].progress_apply(lambda x: model.encode(x))
    train_df = df[df['train'] == True]
    test_df = df[df['train'] == False]

    preprocess = Preprocess(vocab_size=args.vocab_size)
    all_docs = df.news.to_list()
    rst = preprocess.preprocess(all_docs)
    vocab = rst['vocab']
    vocab_dict = {}
    for i, v in enumerate(vocab):
        vocab_dict[v] = i

    train_docs = train_df.news.to_list()
    train_parsed_docs, train_bow = preprocess.parse(train_docs, vocab=vocab)
    train_bow_array = train_bow.toarray()
    test_docs = test_df.news.to_list()
    test_parsed_docs, test_bow = preprocess.parse(test_docs, vocab=vocab)
    test_bow_array = test_bow.toarray()

    train_bow_check = np.where(train_bow_array.sum(axis=0)==0, 0, 1)
    test_bow_check = np.where(test_bow_array.sum(axis=0)==0, 0, 1)
    bow_check = np.where((train_bow_check + test_bow_check)==2, 1, 0)
    bow_check_list = []
    for i in bow_check.argsort():
        if bow_check[i] == 0:
            bow_check_list.append(i)
    bow_check_list.sort()
    for i, v in enumerate(bow_check_list):
        train_bow_array = np.delete(train_bow_array, v - i, axis = 1)
        test_bow_array = np.delete(test_bow_array, v - i, axis = 1)

    sorted_by_value = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=False)
    vocab_dict = dict(sorted_by_value)
    mi_list = bow_check_list
    mi = 0
    new_vocab_dict = dict()
    for w in vocab_dict:
        num = vocab_dict[w]
        if num in mi_list:
            mi += 1
        else:
            new_vocab_dict[w] = num - mi
    print('len vocab :', len(new_vocab_dict))

    check_vocab = [i for i in new_vocab_dict.keys()]
    train_parsed_news, check_bow = preprocess.parse(train_docs, vocab=check_vocab)
    chekc_bow_array = check_bow.toarray()
    check_result = np.sum(np.abs(chekc_bow_array - train_bow_array))
    print('train bow check :', check_result)
    test_parsed_news, check_bow = preprocess.parse(test_docs, vocab=check_vocab)
    chekc_bow_array = check_bow.toarray()
    check_result = np.sum(np.abs(chekc_bow_array - test_bow_array))
    print('test bow check :', check_result)

    train_label = train_df.target.to_list()
    test_label = test_df.target.to_list()
    train_emb = train_df.emb.to_list()
    test_emb = test_df.emb.to_list()
    train_news = train_df.news.to_list()
    test_news = test_df.news.to_list()

    tb1 = np.sum(train_bow_array, axis=1)
    tb1 = np.where(tb1 <= 1, 1, 0)
    tb1_check = tb1.sum()
    if tb1_check != 0:
        index = [ind for ind in tb1.argsort() if tb1[ind] == 1]
        train_bow_array = np.delete(train_bow_array, index, axis=0)
        train_label = np.delete(train_label, index)
        train_emb = np.delete(train_emb, index, axis=0)
        train_news = np.delete(train_news, index)

    tb2 = np.sum(test_bow_array, axis=1)
    tb2 = np.where(tb2 <= 1, 1, 0)
    tb2_check = tb2.sum()
    if tb2_check != 0:
        index = [ind for ind in tb2.argsort() if tb2[ind] == 1]
        test_bow_array = np.delete(test_bow_array, index, axis=0)
        test_label = np.delete(test_label, index)
        test_emb = np.delete(test_emb, index, axis=0)
        test_news = np.delete(test_news, index)

    if train_bow_array.sum(axis=0).min() == 0:
        raise AssertionError('train 0 word error')
    else:
        print('train 0 word check')

    if test_bow_array.sum(axis=0).min() == 0:
        raise AssertionError('test 0 word error')
    else:
        print('test 0 word check')

    print('save data...')
    output_dict = {}
    output_dict['train_bow'] = train_bow_array
    output_dict['test_bow'] = test_bow_array
    output_dict['train_label'] = train_label
    output_dict['test_label'] = test_label
    output_dict['train_emb'] = train_emb
    output_dict['test_emb'] = test_emb
    output_dict['topic_names'] = data.target_names
    output_dict['vocab_dict'] = new_vocab_dict
    output_dict['train_news'] = train_news
    output_dict['test_news'] = test_news
    output_dict['train_parsed_news'] = train_parsed_news
    output_dict['test_parsed_news'] = test_parsed_news

    with open('./dataset/20ng.pickle','wb') as fw:
        pickle.dump(output_dict, fw)


if __name__ == '__main__':
    main()