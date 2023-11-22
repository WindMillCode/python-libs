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
    model_file_name="AmazonReviewFull.pkl",
    dataloader_info ={
      "datapipe_fn":datasets.AmazonReviewFull,
      "vocab_folder_path":"data/AmazonReviewFull",
      "get_dataset":False
    },
  )

myai.download_train_and_test_data()
myai.load_model_from_file()
myai.chat_with_model(500)



