


import itertools
import random
import torchdata


class WMLDataset():

  def __init__(self,datapipe):
    self.datapipe = datapipe
    self.datapipe_as_list = list(self.datapipe)
    self.dataset_size = 0
    self.chars = ""
    vocab = set()
    self.full_data =" ".join(list(map(lambda x:x[1],self.datapipe_as_list)))

    for label, line in self.datapipe:
      characters = set(line)
      vocab.update(characters)
      # self.full_data += line

    self.dataset_size = len(self.full_data)
    self.chars = sorted(vocab)
    self.vocab_size = len(self.chars)

  def get_random_chunk(self,chunk_size):
    start_pos = random.randint(
      0, self.dataset_size - chunk_size)
    random_chunk = self.full_data[start_pos:start_pos+chunk_size]

    return random_chunk
