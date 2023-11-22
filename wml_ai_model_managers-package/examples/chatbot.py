import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse

from wml_ai_model_managers.text_model_manager_one.model_manager import WMLTextModelManagerOne
from wml_ai_model_managers.text_model_manager_one.dataset import WMLDataset
from torchtext import datasets



myai =  WMLTextModelManagerOne(
    model_name="AmazonReviewFull.pkl",
    training_dataloader= WMLDataset(
      datapipe=datasets.AmazonReviewFull(
        split="train"
      )
    ),
    test_dataloader= WMLDataset(
      datapipe=datasets.AmazonReviewFull(
        split="test"
      )
    )
  )

myai.download_train_and_test_data()
myai.load_model_from_file()
myai.chat_with_model()



