


import itertools
import random
import torch
import torchdata


class WMLDataset():

  def __init__(self,datapipe,**kwargs):

    self.device =  kwargs.get("device",'cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading dataset into dataloader")
    self.datapipe = datapipe
    self.datapipe_as_list = list(self.datapipe)
    self.dataset_size = 0
    self.chars = ""

    self.full_data =" ".join(list(map(self.pull_sentence_from_tuple,self.datapipe_as_list)))
    vocab = set(self.full_data)
    self.dataset_size = len(self.full_data)
    self.chars = sorted(vocab)
    self.vocab_size = len(self.chars)

  def get_random_chunk(self,chunk_size):
    start_pos = random.randint(
      0, self.dataset_size - chunk_size)
    random_chunk = self.full_data[start_pos:start_pos+chunk_size]

    return random_chunk

  def pull_sentence_from_tuple(self,input_tuple):

    target_list = list(input_tuple)
    target_item = list(filter(lambda item:isinstance(item, str) and len(item.split()) > 1  ,target_list))

    return target_item[0]
