import os
import threading
from wml_ai_model_managers.text_model_manager_one.model_manager import WMLTextModelManagerOne
from wml_ai_model_managers.text_model_manager_one.dataset import WMLDataset

from wml_ai_model_managers.text_model_manager_zero.model_manager import WMLTextModelManagerZero
from wml_ai_model_managers.vision_model_manager_zero.model_manager import WMLVisionModelManager0
from torchtext import datasets



def train_with_text_data_v1(myai=None):
  myai = myai if myai else WMLTextModelManagerOne(
    model_file_name="AmazonReviewFull.pkl",
    dataloader_info ={
      "datapipe_fn":datasets.AmazonReviewFull,
      "vocab_folder_path":"data/AmazonReviewFull",
      "get_dataset":True
    }
  )

  myai.download_train_and_test_data()
  # myai.load_model_from_scratch()
  myai.load_model_from_pickle_file()
  myai.train()
  myai.save_model_to_pickle_file()


if __name__ == '__main__':
  train_with_text_data_v1()
