import os
from wml_ai_model_managers.text_model_manager_one.model_manager import WMLTextModelManagerOne

from wml_ai_model_managers.text_model_manager_zero.model_manager import WMLTextModelManagerZero
from wml_ai_model_managers.vision_model_manager_zero.model_manager import WMLVisionModelManager0



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

def train_with_text_data_v1():
  myai = WMLTextModelManagerOne()

  myai.download_train_and_test_data()
  myai.get_vocab_info()
  myai.get_encoders()
  myai.get_model_from_scratch()
  myai.train()
  myai.estimate_loss()
  myai.create_optimizer()
  myai.save_model()

if __name__ == '__main__':
  train_with_text_data_v1()
