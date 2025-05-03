import numpy as np
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans


def get_topic_diversity(topics, num_topic, top_k):
    topics = topics.cpu().detach().numpy()
    list_w = np.zeros((num_topic, top_k))
    for i in range(num_topic):
        idx = topics[i,:].argsort()[-top_k:][::-1]
        list_w[i,:] = idx
    n_unique = len(np.unique(list_w))
    td = n_unique / (top_k * num_topic)
    return td


def get_topic_coherence(topics, top_k, parsed_docs, word2id):
    id2word = dict()
    for w in word2id:
        id2word[word2id[w]] = w
    corpora_dict = corpora.Dictionary()
    corpora_dict.token2id = word2id
    corpora_dict.id2token = id2word
    top_words = [[id2word[i.cpu().item()] for i in topic.argsort(descending=True)[:top_k]] for topic in topics]
    parsed_docs = [i.split(' ') for i in parsed_docs]
    coherence_model = CoherenceModel(topics=top_words, texts=parsed_docs, dictionary=corpora_dict, coherence='c_v')
    cv_score = coherence_model.get_coherence()
    return cv_score


def compute_purity_nmi(num_topics, y_true, y_pred):
    contingency_matrix = np.zeros((num_topics, len(set(y_true))))
    for i, label in enumerate(y_pred):
        contingency_matrix[label, y_true[i]] += 1
    max_sum = np.sum(np.amax(contingency_matrix, axis=1))
    purity_score = max_sum / len(y_true)
    nmi_score = normalized_mutual_info_score(y_true, y_pred)
    return purity_score, nmi_score


def compute_km_purity_nmi(num_topics, y_true, y_emb):
    np_y_emb = np.squeeze(np.array(y_emb), axis=None)
    kmeans = KMeans(n_clusters=num_topics, random_state=42, n_init="auto").fit(np_y_emb)
    purity_score, nmi_score = compute_purity_nmi(num_topics, y_true, kmeans.labels_)
    return purity_score, nmi_score