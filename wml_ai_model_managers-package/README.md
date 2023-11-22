A package with several AI classes making it easy to train data based on PyTorch.

## Initialization

To initialize:

```python
# Make sure the test and train data come from the same dataset
myai = WMLTextModelManagerOne(
    model_name="AmazonReviewFull.pkl",
    training_dataloader=WMLDataset(
        datapipe=datasets.AmazonReviewFull(
            split="train"
        )
    ),
    test_dataloader=WMLDataset(
        datapipe=datasets.AmazonReviewFull(
            split="test"
        )
    )
)
```

## Training

To train:

```python
myai.download_train_and_test_data()
myai.load_model_from_scratch()
myai.train()
myai.estimate_loss()
myai.create_optimizer()
myai.save_model_to_pickle()
```

## Chat

To chat:

```python
myai.download_train_and_test_data()
myai.load_model_from_file()
myai.chat_with_model()
```

## Class Initialization Properties

- `device`: Specifies whether to use CUDA, CPU, or lets it be determined by available computer hardware.
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

