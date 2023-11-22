import os
import threading
from wml_ai_model_managers.text_model_manager_one.model_manager import WMLTextModelManagerOne
from wml_ai_model_managers.text_model_manager_one.dataset import WMLDataset

from wml_ai_model_managers.text_model_manager_zero.model_manager import WMLTextModelManagerZero
from wml_ai_model_managers.vision_model_manager_zero.model_manager import WMLVisionModelManager0
from torchtext import datasets


def train_with_random_train_and_test_data():
  my_manager = WMLVisionModelManager0()
  my_manager._get_vocab_info(
    os.path.join("examples","vocab.txt")
  )
  my_manager.get_encoders()
  my_manager.get_model_from_scratch()

  my_manager.train()
  my_manager.estimate_loss()
  my_manager.create_optimizer()
  my_manager.save_model_via_pickle()

def train_with_text_data_v0():
  my_manager = WMLTextModelManagerZero()
  my_manager.download_train_and_test_data()
  my_manager.create_data_loaders()
  my_manager.get_vocab_info()
  my_manager.get_device()
  my_manager.get_model_from_scratch()
  my_manager.place_model_on_device()
  my_manager.create_loss_fn()
  my_manager.create_optimizer()
  my_manager.run_training_process()
  my_manager.save_model_via_pytorch()

def train_with_text_data_v1(myai=None):
  myai = myai if myai else WMLTextModelManagerOne(
    model_name="AmazonReviewFull.pkl",
    dataloader_info ={
      "datapipe_fn":datasets.AmazonReviewFull,
      "vocab_folder_path":"data/AmazonReviewFull",
      "get_dataset":True
    },
    # training_dataloader= WMLDataset(
    #   datapipe=datasets.AmazonReviewFull(
    #     split="train"
    #   )
    # ),
    # test_dataloader= WMLDataset(
    #   datapipe=datasets.AmazonReviewFull(
    #     split="test"
    #   )
    # )
  )

  myai.download_train_and_test_data()
  myai.load_model_from_scratch()
  myai.train()
  myai.estimate_loss()
  myai.create_optimizer()
  myai.save_model_to_pickle()

def train_with_several_datasets():
  myais = [
    # WMLTextModelManagerOne(
    # model_name="CNNDM.pkl",
    #   training_dataloader= WMLDataset(
    #     datapipe=datasets.CNNDM(
    #       split="train"
    #     )
    #   ),
    #   test_dataloader= WMLDataset(
    #     datapipe=datasets.CNNDM(
    #       split="test"
    #     )
    #   )
    # ),
    # WMLTextModelManagerOne(
    #   model_name="AmazonReviewPolarity.pkl",
    #   training_dataloader= WMLDataset(
    #     datapipe=datasets.AmazonReviewPolarity(
    #       split="train"
    #     )
    #   ),
    #   test_dataloader= WMLDataset(
    #     datapipe=datasets.AmazonReviewPolarity(
    #       split="test"
    #     )
    #   )
    # ),
    WMLTextModelManagerOne(
      model_name="CoLA.pkl",
      training_dataloader= WMLDataset(
        datapipe=datasets.CoLA(
          root="data",
          split="train"
        )
      ),
      test_dataloader= WMLDataset(
        datapipe=datasets.CoLA(
          root="data",
          split="test"
        )
      )
    ),
    # WMLTextModelManagerOne(
    #   model_name="CoNLL2000Chunking.pkl",
    #   training_dataloader= WMLDataset(
    #     datapipe=datasets.CoNLL2000Chunking(
    #       split="train"
    #     )
    #   ),
    #   test_dataloader= WMLDataset(
    #     datapipe=datasets.CoNLL2000Chunking(
    #       split="test"
    #     )
    #   )
    # )
  ]

  for myai in myais:
    # train_with_text_data_v1(myai)
    my_thread = threading.Thread(
      target=train_with_text_data_v1,
      args=[
        myai
      ],
      daemon=True
    )
    my_thread.start()


if __name__ == '__main__':
  # train_with_several_datasets()
  train_with_text_data_v1()
