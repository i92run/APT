import torch
import torch.nn.functional as F


class SwiGLU(torch.nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


def get_activation(act):
    if act == 'tanh':
        act = torch.nn.Tanh()
    elif act == 'relu':
        act = torch.nn.ReLU()
    elif act == 'softplus':
        act = torch.nn.Softplus()
    elif act == 'sigmoid':
        act = torch.nn.Sigmoid()
    elif act == 'leakyrelu':
        act = torch.nn.LeakyReLU()
    elif act == 'elu':
        act = torch.nn.ELU()
    elif act == 'selu':
        act = torch.nn.SELU()
    elif act == 'silu':
        act = torch.nn.SiLU()
    elif act == 'gelu':
        act = torch.nn.GELU()
    elif act == 'glu':
        act = torch.nn.GLU()
    elif act == 'swiglu':
        act = SwiGLU()
    else:
        print('Defaulting to tanh activations...')
        act = torch.nn.Tanh()
    return act


class APT(torch.nn.Module):
    def __init__(self, emb_size, num_hiddens, vocab_size, act):
        super(APT, self).__init__()
        self.emb_size = emb_size
        self.num_hiddens = num_hiddens
        self.vocab_size = vocab_size
        self.activation = get_activation(act)
        if act == 'glu' or act == 'swiglu':
            self.model_num_hiddens = int(num_hiddens * 2)
        else:
            self.model_num_hiddens = num_hiddens
        self.hidden_layers = torch.nn.Sequential(
            torch.nn.Linear(self.emb_size, self.model_num_hiddens),
            torch.nn.BatchNorm1d(self.model_num_hiddens),
            self.activation,
            torch.nn.Linear(self.num_hiddens, self.model_num_hiddens),
            torch.nn.BatchNorm1d(self.model_num_hiddens),
            self.activation)
        self.mu_dense = torch.nn.Linear(self.num_hiddens, self.num_hiddens, bias=True)
        self.logvar_dense = torch.nn.Linear(self.num_hiddens, self.num_hiddens, bias=True)
        self.topic_embeddings = None
        self.word_embeddings = torch.nn.Parameter(torch.randn(self.vocab_size, self.num_hiddens).cuda())
        torch.nn.init.kaiming_normal_(self.word_embeddings, mode='fan_out')

    def set_topic_embeddings(self, topic_embeddings):
        self.topic_embeddings = topic_embeddings

    def get_topic_embeddings(self):
        if self.topic_embeddings is not None:
            normalized_topic_embeddings = self.topic_embeddings / torch.sqrt(self.topic_embeddings.pow(2).sum(dim=1, keepdim=True))
            return normalized_topic_embeddings
        else:
            print('Set topic embeddings')

    def get_word_embeddings(self):
        normalized_word_embeddings = self.word_embeddings / torch.sqrt(self.word_embeddings.pow(2).sum(dim=1, keepdim=True))
        return normalized_word_embeddings

    def get_topics(self):
        if self.topic_embeddings is not None:
            topic_word_cos = F.linear(self.get_topic_embeddings(), self.get_word_embeddings())
            topic_word_softmax = F.softmax(topic_word_cos, dim=1)
            return topic_word_softmax
        else:
            print('Set topic embeddings')

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def forward(self, doc_embedding):
        emb = self.hidden_layers(doc_embedding)
        mu = self.mu_dense(emb)
        logvar = self.logvar_dense(emb)
        z = self.reparameterize(mu, logvar)
        z_norm = z / torch.sqrt(z.pow(2).sum(dim=1, keepdim=True))
        return z_norm, mu, logvar


class apt_loss(torch.nn.Module):
    def __init__(self, num_topics, num_hiddens, vocab_size, alpha = 64.0, mrg = 0.1):
        torch.nn.Module.__init__(self)
        self.num_topics = num_topics
        self.num_hiddens = num_hiddens
        self.vocab_size = vocab_size
        self.eye = torch.eye(self.vocab_size).to('cuda')
        self.alpha = alpha
        self.mrg = mrg
        self.topic_embeddings = torch.nn.Parameter(torch.randn(self.num_topics, self.num_hiddens).cuda())
        torch.nn.init.kaiming_normal_(self.topic_embeddings, mode='fan_out')

    def adaptive_pseudo_labeling(self, labels_tf, word_embeddings_norm, topic_embeddings_norm):
        b_size = labels_tf.shape[0]
        labels_p = labels_tf.sum(dim=0, keepdim=True) / b_size
        labels_p = labels_p.repeat(self.vocab_size, 1)
        PiPj = labels_p * labels_p.T
        Pij = torch.mm(labels_tf.T, labels_tf) / b_size
        npmi = (torch.log(Pij / PiPj) / torch.log(Pij)) * -1.
        npmi = torch.where(Pij==0, -1., npmi)
        npmi = torch.where(Pij==1, 0., npmi)
        npmi_exp = torch.exp(npmi)
        npmi_exp = torch.where(PiPj==0, 0., npmi_exp)
        npmi_exp = torch.where(self.eye == 1, 0., npmi_exp)
        doc_npmi = torch.mm(labels_tf, npmi_exp)
        doc_npmi = torch.where(labels_tf==0, 0., doc_npmi)
        topic_word_cos = F.linear(topic_embeddings_norm, word_embeddings_norm)
        topic_word_softmax = F.softmax(topic_word_cos, dim=1)
        doc_topic_distribution = torch.mm(doc_npmi, topic_word_softmax.T)

        pseudo = doc_topic_distribution.max(dim=1, keepdim=True)[0] == doc_topic_distribution
        pseudo_tf = torch.where(pseudo==True, 1., 0.)
        pseudo_tf = pseudo_tf.sum(dim=0, keepdim=True)
        pseudo_tf = torch.where(pseudo_tf==0, True, False)
        pseudo_tf = pseudo_tf.repeat(b_size, 1)
        pseudo_labels = pseudo | pseudo_tf
        return pseudo_labels

    def forward(self, z_norm, labels, word_embeddings_norm):
        labels_tf = torch.where(labels==0, 0., 1.)
        topic_embeddings_norm = self.topic_embeddings / torch.sqrt(self.topic_embeddings.pow(2).sum(dim=1, keepdim=True))
        adaptive_pseudo_labels = self.adaptive_pseudo_labeling(labels_tf, word_embeddings_norm, topic_embeddings_norm)

        doc_topic_cos = F.linear(z_norm, topic_embeddings_norm)
        pos_exp_dt = torch.exp(-self.alpha * (doc_topic_cos - self.mrg))
        neg_exp_dt = torch.exp(self.alpha * (doc_topic_cos + self.mrg))
        p_sim_sum_dt = torch.where(adaptive_pseudo_labels==True, pos_exp_dt, 0.).sum(dim=0)
        n_sim_sum_dt = torch.where(adaptive_pseudo_labels==False, neg_exp_dt, 0.).sum(dim=0)
        pos_term_dt = torch.log(1 + p_sim_sum_dt).sum()
        neg_term_dt = torch.log(1 + n_sim_sum_dt).sum()
        loss_dt = (pos_term_dt + neg_term_dt) / self.num_topics

        doc_word_cos = F.linear(z_norm, word_embeddings_norm)
        n_labels = 1 - labels_tf
        pos_exp_dw = torch.where(labels_tf == 1, torch.exp(-doc_word_cos), 0.)
        neg_exp_dw = torch.where(n_labels == 1, torch.exp(doc_word_cos), 0.)
        p_sim_sum_dw = pos_exp_dw.sum(dim=1, keepdim=True)
        n_sim_sum_dw = neg_exp_dw.sum(dim=1, keepdim=True)
        pos_term_dw = torch.log(1 + p_sim_sum_dw)
        neg_term_dw = torch.log(1 + n_sim_sum_dw)
        loss_dw = pos_term_dw + neg_term_dw
        loss_dw = loss_dw.mean()

        return loss_dt, loss_dw


def kl_div(mu, logvar):
    loss_kl = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar, dim=-1)
    return loss_kl.mean()