


import itertools
import random
import torch
import torchdata


class WMLDataset():

  def __init__(self,datapipe,**kwargs):

    self.device =  kwargs.get("device",'cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading data set into dataloader")
    self.datapipe = datapipe
    self.datapipe_as_list = list(self.datapipe)
    self.dataset_size = 0
    self.chars = ""
    self.full_data =" ".join(list(map(lambda x:x[1],self.datapipe_as_list)))
    vocab = set(self.full_data)
    self.dataset_size = len(self.full_data)
    self.chars = sorted(vocab)
    self.vocab_size = len(self.chars)

  def get_random_chunk(self,chunk_size):
    start_pos = random.randint(
      0, self.dataset_size - chunk_size)
    random_chunk = self.full_data[start_pos:start_pos+chunk_size]

    return random_chunk
