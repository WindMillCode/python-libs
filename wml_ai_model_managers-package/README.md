# WML AI Model Managers

A package with several AI classes making it easy to train data based on PyTorch.

## Initialization

To initialize:

```python
# Make sure the test and train data come from the same dataset
myai = WMLTextModelManagerOne(
    model_file_name="AmazonReviewFull.pkl",
    dataloader_info ={
      "datapipe_fn":datasets.AmazonReviewFull,
      "vocab_folder_path":"data/AmazonReviewFull",
      "get_dataset":True
    }
  )
```

## Training

To train:

```python
  myai.download_train_and_test_data()
  myai.load_model_from_scratch()
  myai.train()
  myai.save_model_to_pickle()
```

## Chat

To chat:

```python
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
# the default is 150 but we increased to 500 the limit should be myai.batch_size * myai.block_size but see if you can get larger than this
myai.chat_with_model(500)
```

## Class Initialization Properties

- `device`: Specifies whether to use CUDA, CPU, or lets it be determined by available computer hardware.
- `dataloader_info`: Important information about the pytorch dataset the model manager needs to retrieve the dataset
    - `datapipe_fn`: One of the various pytorch datapipes which can be found here https://pytorch.org/text/stable/datasets.html. You can implement your own custom datapipe fn as the model is looking for an IterDataPipe, make sure your custom fn have train and test splits.
    - `vocab_folder_path`: Where the vocab file for the dataset will be stored. it will store test-vocab.txt and train-vocab.txt.
    - `get_dataset`: Set to false if communicating with chatbot set to true if doing a training session.
- `max_iters`: Number of iterations for the entire training run.
- `n_embd`: Number of embeddings.
- `n_head`: Number of heads for multihead attention.
- `n_layer`: Number of layers (linear, activation, output functions).
- `dropout`: Number of values to turn to zero to prevent the model from memorization.
- `model_file_name`: The name of the file to save the model to (currently supports only pickle files, so please save as .pkl).
- `reporting_loss`.
- `learning_rate`.
- `block_size`: Amount of characters in a section of text that represents 1 batch.
- `batch_size`: Amount of batches the model gets to learn in 1 iteration during the training session. A training session is like a child going through pre-K through college, and each grade is 1 iteration. At the end, the model should be able to generalize (converge) well, like an adult who can meaningfully contribute to society.

## Changelog

### v0.0.2:

- Corrected issues with the dataloader where the model_manager would not receive the training and test dataloaders.

### v0.0.3:

- Changed the default block and batch sizes so beginners can feel more tangible results.
- Refactored

### v1.0.0:
- instead of providing a dataloader argument to the modelManger provide dataloader_info according to the example above

### v1.0.1
* abstracted more fns into

### v1.1.0
* changed from load_model_from_file to load_model_from_pickle_file
* changed from save_model_to_pickle to save_model_to_pickle_file

### v1.1.1
* fixed a bug with v1.1.0
