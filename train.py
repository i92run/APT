from utils.train_utils import *
from utils.eval_utils import get_topic_diversity, get_topic_coherence, compute_purity_nmi, compute_km_purity_nmi
from model.apt import APT, apt_loss, kl_div
import argparse
import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np


parser = argparse.ArgumentParser(description='APT_train')
parser.add_argument('--seed', type=int, default=51, help='random seed')
parser.add_argument('--data_path', type=str, default='./dataset/20ng.pickle', help='dataset load path')
parser.add_argument('--save_path', type=str, default='./output', help='model save path')
parser.add_argument('--topic', type=int, default=50, help='num topics')
parser.add_argument('--tc_topk', type=int, default=15, help='top words for topic coherence')
parser.add_argument('--td_topk', type=int, default=15, help='top words for topic diversity')
parser.add_argument('--hs', type=int, default=512, help='hidden size')
parser.add_argument('--act', type=str, default='silu', help='activation')
parser.add_argument('--epochs', type=int, default=300, help='epoch num')
parser.add_argument('--bs', type=int, default=512, help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--wd', type=float, default=1e-6, help='weight decay')
parser.add_argument('--alpha', type=float, default=64., help='hyperparameter alpha')
parser.add_argument('--mrg', type=float, default=0.1, help='hyperparameter margin')
args = parser.parse_args()


def main():
    set_seed(args.seed, True)

    if not os.path.isdir(args.save_path):
        os.system('mkdir -p ' + args.save_path)

    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)
    train_dataset = APT_Dataset(data['train_emb'], data['train_bow'])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, drop_last=True)

    emb_size = data['train_emb'][0].shape[0]
    word2id = data['vocab_dict']
    vocab_size = len(word2id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = APT(emb_size, args.hs, vocab_size, args.act).to(device)
    criterion = apt_loss(args.topic, args.hs, vocab_size, args.alpha, args.mrg).to(device)
    param_groups = [{'params': model.parameters()}]
    param_groups.append({'params': criterion.parameters(), 'lr':float(args.lr)*100.})
    optimizer = torch.optim.Adam(param_groups, lr=args.lr, weight_decay=args.wd)

    best_td = 0.
    for epoch in range(args.epochs):
        model.train()
        loss_list, loss_kl_list, loss_dw_list, loss_dt_list = [], [], [], []
        for idx, batch in enumerate(train_loader):
            inputs = batch[0].to(device)
            labels = batch[1].to(device)

            z_norm, mu, logvar = model(inputs)
            word_embeddings_norm = model.get_word_embeddings()
            loss_dt, loss_dw = criterion(z_norm, labels, word_embeddings_norm)
            loss_kl = kl_div(mu, logvar)
            loss = loss_kl + loss_dw + loss_dt

            loss_list.append(loss.item())
            loss_kl_list.append(loss_kl.item())
            loss_dw_list.append(loss_dw.item())
            loss_dt_list.append(loss_dt.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch: [{}/{}] \tLoss: {:.4f} \tKL: {:.4f} \tDW: {:.4f} \tDT: {:.4f}'.format(
            epoch + 1, args.epochs, np.mean(loss_list),
            np.mean(loss_kl_list), np.mean(loss_dw_list), np.mean(loss_dt_list))
        )

        model.set_topic_embeddings(criterion.topic_embeddings)
        model.eval()
        with torch.no_grad():
            now_topics = model.get_topics()
            now_td = get_topic_diversity(now_topics, args.topic, args.td_topk)
            if now_td >= best_td:
                best_td = now_td
                torch.save(model.state_dict(), args.save_path + '/model_weights.pth')

    model.load_state_dict(torch.load(args.save_path + '/model_weights.pth'))
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
            temp_cos = []
            for idx, batch in enumerate(full_test_loader):
                inputs = batch[0].to(device)
                z_norm, mu, logvar = model(inputs)
                cos = F.linear(z_norm, topics_embeddings_norm)
                temp_cos.append(cos.cpu())
            test_doc_topic_cos = torch.cat(temp_cos)

        test_doc_topic_softmax = torch.nn.functional.softmax(test_doc_topic_cos, dim=-1)
        test_pred = np.array(test_doc_topic_softmax)
        test_pred = np.squeeze(test_pred, axis=None)
        test_pred = np.argmax(test_pred, axis=1)

        top_purity, top_nmi = compute_purity_nmi(args.topic, data['test_label'], test_pred)
        print('Top-Purity: {:.4f}, Top-NMI: {:.4f}'.format(top_purity, top_nmi))

        km_purity, km_nmi = compute_km_purity_nmi(args.topic, data['test_label'], test_doc_topic_softmax)
        print('KM-Purity: {:.4f}, KM-NMI: {:.4f}'.format(km_purity, km_nmi))


if __name__ == '__main__':
    main()