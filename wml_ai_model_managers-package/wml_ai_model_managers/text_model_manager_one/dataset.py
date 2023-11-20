


import itertools
import os
import random
import torch
import torchdata
from importlib import import_module
from torchtext.data.datasets_utils import _create_dataset_directory,_wrap_split_argument
from tqdm import tqdm
from wml_ai_model_managers.wml_utils.common_utils import find_file

class WMLDataset():

  loaded_dataset_from_datapipe = False
  def __init__(self,**kwargs):


    self.dataloader_info =  kwargs.get("dataloader_info")
    self.split = kwargs.get("split")
    self.extract_text_predicate = kwargs.get("extract_text_predicate",self.pull_sentences_from_tuple)
    self.get_target_pytorch_dataset_file =  self.dataloader_info.get("get_target_pytorch_dataset_file",lambda x:x+".csv")
    self.get_target_pytorch_text_file =  self.dataloader_info.get("get_target_pytorch_text_file",lambda x:x+".text")

    dataset_directory = self.get_dataset_directory()
    target_csv_files = find_file(
      self.get_target_pytorch_dataset_file(self.split),dataset_directory
    )
    target_text_files = find_file(
      self.get_target_pytorch_text_file(self.split),dataset_directory
    )

    if len(target_text_files)<1:
      datapipe_fn = self.dataloader_info.get("datapipe_fn")
      root = self.dataloader_info.get("root")
      datapipe = datapipe_fn(root=root,split=self.split)
      self.get_dataset_from_datapipe(datapipe)
      self.dataset_file =self.get_dataset_file(dataset_directory)
      print(self.full_data[0:100])
      with open(self.dataset_file,"w",encoding="utf-8") as outfile:
        for char in tqdm(self.full_data,total=len(self.full_data)):
            outfile.write(char)

    else:
      self.dataset_file =self.get_dataset_file(dataset_directory)
      self.get_dataset_from_file()


  def get_dataset_file(self, dataset_directory):
      target_csv_files = find_file(self.get_target_pytorch_dataset_file(self.split),dataset_directory)
      csv_file = target_csv_files[0]
      return os.path.join(
        os.path.dirname(csv_file),
        self.get_target_pytorch_text_file(self.split)
      )



  def get_dataset_directory(self):
      datapipe_fn = self.dataloader_info.get("datapipe_fn")
      root = self.dataloader_info.get("root")

      dataset_module = import_module(datapipe_fn.__module__)
      filepath_fn = getattr(dataset_module,"_filepath_fn",None)
      DATASET_NAME = getattr(dataset_module,"DATASET_NAME",None)

      dataset_directory = _create_dataset_directory(
        DATASET_NAME
      )(filepath_fn)(root)
      dataset_directory =os.path.dirname(dataset_directory)
      return dataset_directory


  def get_dataset_from_file(self):
      with open(self.dataset_file,"r",encoding="utf-8") as outfile:

        self.full_data =  outfile.read()
        self.dataset_size = 0
        vocab = set(self.full_data)
        self.dataset_size = len(self.full_data)
        self.chars = sorted(vocab)
        self.vocab_size = len(self.chars)

  def get_dataset_from_datapipe(self, datapipe):
      self.datapipe = datapipe
      self.datapipe_as_list = list(self.datapipe)
      self.dataset_size = 0
      self.full_data =" ".join(list(map(self.extract_text_predicate,self.datapipe_as_list)))
      vocab = set(self.full_data)
      self.dataset_size = len(self.full_data)
      self.chars = sorted(vocab)
      self.vocab_size = len(self.chars)
      self.loaded_dataset_from_datapipe= True
      self.datapipe_as_list =[]

  def get_random_chunk(self,chunk_size):
    start_pos = random.randint(
      0, self.dataset_size - chunk_size)
    random_chunk = self.full_data[start_pos:start_pos+chunk_size]

    return random_chunk

  def pull_sentences_from_tuple(self,input_tuple):

    target_list = list(input_tuple)
    target_item = list(filter(lambda item:isinstance(item, str) and len(item.split()) > 1  ,target_list))

    return " ".join(target_item)
