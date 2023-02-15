import torch
import torch.nn as nn

'''
attempt of implementing self attention from scratch
'''

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data
        self.words = [el.replace('\n', '') for el in data.split(' ')]

    def load_list(self):
        corpus = []





class Model(nn.Module):
    def __init__(self,word_list, emb_dim=2):
        super().__init__()
        self.w = None
        self.b = None
        
        self.embs = nn.Embedding(len(set(word_list)), emb_dim)

    def encode(self, word):
        pass
    def decode(self, word):
        pass




dataset = MyDataset('eu gosto de café\n com açucar')
print(dataset.words) 