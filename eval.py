from utils.train_utils import APT_Dataset
from model.apt import APT, apt_loss
from utils.eval_utils import *
import argparse
import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np


parser = argparse.ArgumentParser(description='APT_eval')
parser.add_argument('--data_path', type=str, default='./dataset/20ng.pickle', help='dataset load path')
parser.add_argument('--model_path', type=str, default='./output/20ng_model_weights.pth', help='model load path')
parser.add_argument('--save_path', type=str, default='./output', help='image save path')
parser.add_argument('--topic', type=int, default=50, help='num topics')
parser.add_argument('--tc_topk', type=int, default=15, help='top words for topic coherence')
parser.add_argument('--td_topk', type=int, default=15, help='top words for topic diversity')
parser.add_argument('--hs', type=int, default=512, help='hidden size')
parser.add_argument('--act', type=str, default='silu', help='activation')
args = parser.parse_args()


def main():
    if not os.path.isdir(args.save_path):
        os.system('mkdir -p ' + args.save_path)

    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)

    emb_size = data['train_emb'][0].shape[0]
    word2id = data['vocab_dict']
    vocab_size = len(word2id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = APT(emb_size, args.hs, vocab_size, args.act).to(device)
    criterion = apt_loss(args.topic, args.hs, vocab_size, 64.0, 0.1).to(device)
    model.set_topic_embeddings(criterion.topic_embeddings)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    topics = model.get_topics()
    cv = get_topic_coherence(topics, args.tc_topk, data['test_parsed_news'], word2id)
    td = get_topic_diversity(topics, args.topic, args.td_topk)
    print('Cv: {:.4f}, TD: {:.4f}'.format(cv, td))

    if 'test_label' in data.keys():
        test_dataset = APT_Dataset(data['test_emb'], data['test_bow'])
        full_test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, drop_last=False)
        topics_embeddings_norm = model.get_topic_embeddings()
        with torch.no_grad():
            temp_embeddings = []
            temp_cos = []
            for idx, batch in enumerate(full_test_loader):
                inputs = batch[0].to(device)
                z_norm, mu, logvar = model(inputs)
                cos = F.linear(z_norm, topics_embeddings_norm)
                temp_embeddings.append(z_norm.cpu())
                temp_cos.append(cos.cpu())
            test_doc_embeddings = torch.cat(temp_embeddings)
            test_doc_topic_cos = torch.cat(temp_cos)

        test_doc_topic_softmax = torch.nn.functional.softmax(test_doc_topic_cos, dim=-1)
        test_pred = np.array(test_doc_topic_softmax)
        test_pred = np.squeeze(test_pred, axis=None)
        test_pred = np.argmax(test_pred, axis=1)

        top_purity, top_nmi = compute_purity_nmi(args.topic, data['test_label'], test_pred)
        print('Top-Purity: {:.4f}, Top-NMI: {:.4f}'.format(top_purity, top_nmi))

        km_purity, km_nmi = compute_km_purity_nmi(args.topic, data['test_label'], test_doc_topic_softmax)
        print('KM-Purity: {:.4f}, KM-NMI: {:.4f}'.format(km_purity, km_nmi))

    topic_embeddings = model.get_topic_embeddings()
    topic_np = np.array(topic_embeddings.cpu().detach())
    doc_np = np.array(test_doc_embeddings.cpu().detach())
    visualize(topic_np, doc_np, args.save_path + '/tSNE.png')


if __name__ == '__main__':
    main()