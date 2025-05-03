import random
import numpy as np
import torch


def set_seed(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class APT_Dataset(torch.utils.data.Dataset):
    def __init__(self, doc_embedding, bow):
        super(APT_Dataset, self).__init__()
        self.doc_embedding = doc_embedding
        self.bow = bow

    def __getitem__(self, idx):
        return self.doc_embedding[idx], self.bow[idx]

    def __len__(self):
        return len(self.doc_embedding)