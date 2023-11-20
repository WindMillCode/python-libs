from operator import xor
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import mmap
import random
import pickle
import argparse

from wml_ai_model_managers.wml_utils.error_utils import _LogicError
from wml_ai_model_managers.vision_model_manager_zero.model import GPTLanguageModel
from wml_ai_model_managers.vision_model_manager_zero.block import Block
from wml_ai_model_managers.vision_model_manager_zero.feedforward import FeedFoward
from wml_ai_model_managers.vision_model_manager_zero.multi_head_attention import MultiHeadAttention
from wml_ai_model_managers.vision_model_manager_zero.head import Head



class WMLVisionModelManager0():
  def __init__(self,
    block_size =  128,
    batch_size = 64,
    max_iters = 15000,
    reporting_loss = 100,
    learning_rate = 3e-4,
    n_embd = 384,
    n_head = 4,
    n_layer = 4,
    dropout = 0.1,
    device = 'cuda' if torch.cuda.is_available() else 'cpu',
    is_data_logged = True
               ) -> None:

    self.block_size = block_size
    self.batch_size = batch_size
    self.max_iters = max_iters
    self.reporting_loss = reporting_loss
    self.learning_rate = learning_rate
    self.n_embd = n_embd
    self.n_head = n_head
    self.n_layer = n_layer
    self.dropout = dropout
    self.device = device
    self.is_data_logged =is_data_logged

  def debug(self,value):
    if self.is_data_logged:
      self.debug(value)



  def get_model_from_scratch(self):
    self.model = GPTLanguageModel(
      vocab_size=self.vocab_size,
      block_size=self.block_size,
      n_embd=self.n_embd,
      n_layer=self.n_layer,
      device=self.device,
      n_head=self.n_head,
      dropout=self.dropout
    )
    self.m = self.model.to(self.device)



  def download_train_and_test_data(self,training_data=None,test_data=None):
      if not xor(training_data,test_data):
        raise _LogicError("if you provide training_data or test_data you must provide the other or else you be testining your training data with a different dataset! If you dont know this is bad!")


      self.training_data = training_data if  training_data else   datasets.AmazonReviewFull(
          root="data",
          train=True,
          download=True,
          transform=ToTensor()
      )

      self.test_data =  test_data if  test_data else  datasets.AmazonReviewFull(
          root="data",
          train=False,
          download=True,
          transform=ToTensor()
      )

  def create_data_loaders(self):
    self.batch_size = 64
    self.train_dataloader = DataLoader(
      self.training_data, batch_size=self.batch_size)
    self.test_dataloader =  DataLoader(
      self.test_data, batch_size=self.batch_size)

    for X, y in self.test_dataloader:
        print(f"Shape of X  {X.shape}")
        print(f"Shape of y {y.shape} {y.dtype}")
        break

  def split_between_train_and_test(self,ratio=0.8):
    n = int(ratio*len(self.data))
    self.train_data = self.data[:n]
    self.val_data = self.data[n:]

  def _get_vocab_info(self,source_file='vocab.txt',encoding='utf-8'):
    self.chars = ""
    with open(source_file,'r',encoding=encoding) as f:
      text = f.read()
      self.chars = sorted(set(text))
    self.vocab_size = len(self.chars)

  def get_encoders(self):
    string_to_int = { ch:i for i,ch in enumerate(self.chars) }
    int_to_string = { i:ch for i,ch in enumerate(self.chars) }
    self.encode = lambda s: [string_to_int[c] for c in s]
    self.decode = lambda l: ''.join([int_to_string[i] for i in l])

  def train(self):
    context = torch.zeros((1,1), dtype=torch.long, device=self.device)
    generated_chars = self.decode(self.m.generate(context, max_new_tokens=500)[0].tolist())

  def get_random_chunk(self,split,filename=None):
    filename = filename if filename else "train_split.txt" if split == 'train' else "val_split.txt"
    with open(filename, 'rb') as f:
      with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        # Determine the file size and a random position to start reading
        file_size = len(mm)
        start_pos = random.randint(
          0, (file_size) - self.block_size*self.batch_size)

        # Seek to the random position and read the block of text
        mm.seek(start_pos)
        block = mm.read(self.block_size*self.batch_size-1)

        # Decode the block to a string, ignoring any invalid byte sequences
        decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')

        # Train and test splits
        return torch.tensor(self.encode(decoded_block), dtype=torch.long)




  def get_batch_via_random_chunk(self,split):
    data = self.get_random_chunk(split)
    return self._get_batch(data=data)

  def get_batch_via_user_tested_data(self,split):
    data = self.train_data if split == "train" else self.val_data
    return self._get_batch(data=data)

  def _get_batch(self,data):
      ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
      x = torch.stack([data[i:i+self.block_size] for i in ix])
      y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
      x, y = x.to(self.device), y.to(self.device)
      return x, y


  @torch.no_grad()
  def estimate_loss(self):
      out = {}
      self.model.eval()
      for split in ['train', 'val']:
          losses = torch.zeros(self.reporting_loss)
          for k in range(self.reporting_loss):
              X, Y = self._get_batch(split)
              logits, loss = self.model(X, Y)
              losses[k] = loss.item()
          out[split] = losses.mean()
      self.model.train()
      return out

  def create_optimizer(self,optimzer_fn=None):
    optimzer_fn = optimzer_fn if optimzer_fn else torch.optim.AdamW
    optimizer = optimzer_fn(self.model.parameters(), lr=self.learning_rate)

    for iter in range(self.max_iters):
        if iter % self.reporting_loss == 0:
            losses = self.estimate_loss()
            self.debug(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")

        # sample a batch of data
        xb, yb = self._get_batch('train')

        # evaluate the loss
        logits, loss = self.model.forward(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return loss.item()


  def load_model_via_pickle(self,file='model-01.pkl'):
    with open(file, 'rb') as f:
        self.model = pickle.load(f)
    self.debug('loaded successfully!')
    self.m = self.model.to(self.device)

  def save_model_via_pickle(self,file='model-01.pkl'):
    with open(file,'wb') as f:
      pickle.dump(self.model,f)
    self.debug('saved successfully!')

  def load_model_via_pytorch(self,target_file='model.pth'):

    self.model = torch.load( target_file)

  def save_model_via_pytorch(self,target_file='model.pth'):
    torch.save(self.model, target_file)


  def chat_with_model(self,max_new_tokens):
    max_new_tokens
    while True:
        prompt = input("Prompt:\n")
        context = torch.tensor(self.encode(prompt), dtype=torch.long, device=self.device)
        generated_chars = self.decode(
          self.m.generate(
            context.unsqueeze(0), max_new_tokens=max_new_tokens)[0].tolist())
        self.debug(f'Completion:\n{generated_chars}')



