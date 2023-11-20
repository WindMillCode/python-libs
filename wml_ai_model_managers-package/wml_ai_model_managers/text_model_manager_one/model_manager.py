import os
import time
import requests
import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse
from tqdm import tqdm

from wml_ai_model_managers.text_model_manager_one.dataset import WMLDataset
from wml_ai_model_managers.text_model_manager_one.model import WMLTextOneModel
from wml_ai_model_managers.text_model_manager_one.block import WMLTextOneBlock
from wml_ai_model_managers.text_model_manager_one.feedforward import WMLTextOneFeedFoward
from wml_ai_model_managers.text_model_manager_one.multi_head_attention import WMLTextOneMultiHead
from wml_ai_model_managers.text_model_manager_one.head import WMLTextOneHead
from wml_ai_model_managers.wml_utils.error_utils import _LogicError
from wml_ai_model_managers.wml_utils.common_utils import xor
from torch.utils.data import DataLoader,Subset
from torchtext.data import datasets_utils

from torchtext import datasets


class WMLTextModelManagerOne():
  def __init__(self,**kwargs) -> None:
    self.block_size =  kwargs.get("block_size",256)
    self.batch_size = kwargs.get("batch_size",256)
    self.device =  kwargs.get("device",'cuda' if torch.cuda.is_available() else 'cpu')
    self.max_iters = kwargs.get("max_iters",15000)
    self.n_embd = kwargs.get("n_embd", 384)
    self.n_head = kwargs.get("n_head", 4)
    self.n_layer = kwargs.get("n_layer", 8)
    self.dropout = kwargs.get("dropout", 0.1)
    self.reporting_loss = kwargs.get("reporting_loss", 100)
    self.learning_rate = kwargs.get("learning_rate", 3e-4)
    self.model_file_name = kwargs.get("model_name",'model-02.pkl')
    self.dataloader_info = kwargs.get("dataloader_info")


  def chat_with_model(self):
    while True:
        prompt = input("Prompt:\n")
        context = torch.tensor(self.encode(prompt), dtype=torch.long, device=self.device)
        generated_chars = self.decode(self.m.generate(context.unsqueeze(0), max_new_tokens=150)[0].tolist())
        print(f'Completion:\n{generated_chars}')

  def load_model_from_scratch(self):
    self.model = WMLTextOneModel(
      vocab_size=self.vocab_size,
      block_size=self.block_size,
      n_embd=self.n_embd,
      n_layer=self.n_layer,
      device=self.device,
      n_head=self.n_head,
      dropout=self.dropout
    )
    self.m = self.model.to(self.device)

  def load_model_from_file(self):
    with open(self.model_file_name, 'rb') as f:
        self.model = pickle.load(f)
    print('loaded successfully!')
    self.m = self.model.to(self.device)

  def save_model(self):
    with open(self.model_file_name,'wb') as f:
      pickle.dump(self.model,f)

  def download_train_and_test_data(self):
      # if xor(training_dataloader,test_dataloader):
      #   raise _LogicError("if you provide training_dataloader or test_dataloader you must provide the other or else you be testining your training data with a different dataset! If you dont know this is bad!")

      self.training_dataloader =  WMLDataset(
        dataloader_info= self.dataloader_info,split="train"
      )

      self.test_dataloader = WMLDataset(
        dataloader_info= self.dataloader_info,split="test"
      )


  def get_vocab_info(self):

    self.chars = sorted(set( self.training_dataloader.chars + self.test_dataloader.chars))
    self.vocab_size = len(self.chars)

  def get_encoders(self):
    string_to_int = { ch:i for i,ch in enumerate(self.chars) }
    int_to_string = { i:ch for i,ch in enumerate(self.chars) }
    self.encode = lambda s: [string_to_int[c] for c in s]
    self.decode = lambda l: ''.join([int_to_string[i] for i in l])


  def get_random_chunk(self,split):
      my_dataset = self.training_dataloader if split == 'train' else self.test_dataloader
      with open(my_dataset.dataset_file, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            start_pos = random.randint(
              0, (file_size) - self.block_size*self.batch_size)

            mm.seek(start_pos)
            block = mm.read(self.block_size*self.batch_size-1)

            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')


            return torch.tensor(self.encode(decoded_block), dtype=torch.long)

  def get_batch(self,split):
      data = self.get_random_chunk(split)
      # data = train_data if split == "train" else val_data
      ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
      x = torch.stack([data[i:i+self.block_size] for i in ix])
      y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
      x, y = x.to(self.device), y.to(self.device)
      return x, y

  def train(self):
    print("Starting Training Session")
    context = torch.zeros((1,1), dtype=torch.long, device=self.device)
    generated_chars = self.decode(self.m.generate(context, max_new_tokens=self.vocab_size)[0].tolist())


  @torch.no_grad()
  def estimate_loss(self):
      out = {}
      self.model.eval()
      for split in ['train', 'val']:
          losses = torch.zeros(self.reporting_loss)
          for k in range(self.reporting_loss):
              X, Y = self.get_batch(split)
              logits, loss = self.model(X, Y)
              losses[k] = loss.item()
          out[split] = losses.mean()
      self.model.train()
      return out

  def create_optimizer(self):
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
    print("Starting Training Session")
    for iter in range(self.max_iters):
        start_time = time.time()
        self.iters_left = self.max_iters - iter
        if iter % self.reporting_loss == 0:
            losses = self.estimate_loss()
            end_time = time.time()
            elapsed_time_seconds = end_time - start_time
            hours, remainder = divmod(elapsed_time_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}, elapsed time: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")

            self.save_model()
        # if iter % 500 == 0:
        # sample a batch of data
        xb, yb = self.get_batch('train')

        # evaluate the loss
        logits, loss = self.model.forward(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return loss.item()





