


import itertools
import os
import random
import torch
import torchdata

from wml_ai_model_managers.wml_utils.common_utils import get_device,create_and_write_to_file

class WMLDataset():

  def __init__(self,datapipe=None,device=None,vocab_file_path=None,dataloader_info=None):

    self.device =  device if device else get_device()
    self._dataloader_info = dataloader_info
    if dataloader_info is not None:
      self.init_based_on_dataloader_info(dataloader_info)

    else:
      self.vocab_file_path = vocab_file_path
      self.init_based_on_dataloader(datapipe)

  def init_based_on_dataloader_info(self, dataloader_info):
      get_dataset = dataloader_info.get("get_dataset",True)
      datapipe_fn = dataloader_info.get("datapipe_fn",None)
      split = dataloader_info.get("split",None)
      self.vocab_file_path = os.path.normpath(
        os.path.join(
          dataloader_info["vocab_folder_path"],
          dataloader_info["split"]+"-vocab.txt"
        )
      )

      if get_dataset == True:
        self.datapipe = datapipe_fn(
          split=split
        )
        self.init_based_on_dataloader(self.datapipe)

      else:
        self.get_vocab_info()


  def init_based_on_dataloader(self, datapipe):

      print("Loading dataset into dataloader")
      self.datapipe = datapipe
      self.datapipe_as_list = list(self.datapipe)
      self.dataset_size = 0
      self.chars = ""

      self.full_data =" ".join(list(map(self.pull_sentence_from_tuple,self.datapipe_as_list)))
      self.dataset_size = len(self.full_data)
      self.get_vocab_info()

  def get_vocab_info(self):
      print("getting vocab info")
      vocab = set()
      try:
        with open(self.vocab_file_path,"r",encoding="utf-8") as f:
          vocab = set(f.read())
      except BaseException as e:

        vocab = set(self.full_data)
      if len(vocab) == 0:
        vocab = set(self.full_data)

      self.chars = sorted(vocab)
      self.vocab_size = len(self.chars)
      create_and_write_to_file(self.vocab_file_path,"".join(self.chars))




  def get_random_chunk(self,chunk_size):
    start_pos = random.randint(
      0, self.dataset_size - chunk_size)
    random_chunk = self.full_data[start_pos:start_pos+chunk_size]

    return random_chunk

  def pull_sentence_from_tuple(self,input_tuple):

    target_list = list(input_tuple)
    target_item = list(filter(lambda item:isinstance(item, str) and len(item.split()) > 1  ,target_list))

    return target_item[0]
