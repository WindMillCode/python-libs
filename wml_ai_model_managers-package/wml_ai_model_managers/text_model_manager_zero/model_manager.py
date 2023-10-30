from operator import xor
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from torchtext import datasets
import mmap
import random
import pickle
import argparse

from model import WMLTextModelZero
from block import WMLBlock
from feedforward import WMLFeedFoward
from multi_head_attention import WMLMultiHeadAttention
from head import WMLHead
from wml_utils.error_utils import _LogicError



class WMLTextModelManagerZero():
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
    self.model = WMLTextModelZero(
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
          download=True
      )

      self.test_data =  test_data if  test_data else  datasets.AmazonReviewFull(
          root="data",
          train=False,
          download=True
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

  def get_device(self,device=None):
    self.device = device if device else (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


  def place_model_on_device(self):
    self.model = WMLTextModelZero().to(self.device)

  def create_loss_fn(self):
      self.loss_fn = nn.CrossEntropyLoss()

  def create_optimizer(self,optimzer_fn=None):
    optimzer_fn = optimzer_fn if optimzer_fn else torch.optim.AdamW
    self.optimizer = optimzer_fn(self.model.parameters(), lr=self.learning_rate)

  def train(self,dataloader=None):
      dataloader = dataloader if dataloader else self.train_dataloader
      size = len(dataloader.dataset)
      self.model.train()
      for batch, (X, y) in enumerate(dataloader):
          X, y = X.to(self.device), y.to(self.device)

          # Compute prediction error
          # pred = self.model(X)
          # loss = self.loss_fn(pred, y)
          logits, loss = self.model.forward(X, y)
          self.optimizer.zero_grad(set_to_none=True)
          loss.backward()
          self.optimizer.step()

          if batch % 100 == 0:
              loss, current = loss.item(), (batch + 1) * len(X)
              print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


  def test(self,dataloader=None):
      dataloader = dataloader if dataloader else self.test_dataloader
      size = len(dataloader.dataset)
      num_batches = len(dataloader)
      self.model.eval()
      test_loss, correct = 0, 0
      with torch.no_grad():
          for X, y in dataloader:
              X, y = X.to(self.device), y.to(self.device)
              pred = self.model(X,y)
              test_loss += self.loss_fn(pred, y).item()
              correct += (pred.argmax(1) == y).type(torch.float).sum().item()
      test_loss /= num_batches
      correct /= size
      print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



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



