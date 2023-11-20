from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'A package for training and communicating with AI chatbots based on pytorch'
LONG_DESCRIPTION = """
=======
WMLText
=======

A package with several AI classes making it easy to train data based on pytorch.

Initialization
--------------

To initialize:

.. code-block:: python

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

Training
--------

To train:

.. code-block:: python

    myai.download_train_and_test_data()
    myai.get_vocab_info()
    myai.get_encoders()
    myai.load_model_from_scratch()
    myai.train()
    myai.estimate_loss()
    myai.create_optimizer()
    myai.save_model()

Chat
----

To chat:

.. code-block:: python

    myai.load_model_from_file()
    myai.chat_with_model()

Class Initialization Properties
-------------------------------

- ``device``: Whether to use cuda, cpu else it is determined by the available computer hardware.
- ``max_iters``: How many iterations for the entire training run.
- ``n_embd``: Number of embeddings.
- ``n_head``: Number of heads for multihead attention.
- ``n_layer``: Number of layers (linear, activation, output fns).
- ``dropout``: Number of values to turn to zero to prevent the model from memorization.
- ``model_file_name``: The name of the file to save the model to (for now only supports pickle files, so please save via .pkl).
- ``reporting_loss``.
- ``learning_rate``.
- ``block_size``.
- ``batch_size``.
"""

# Now you can use the variable 'long_description' as needed in your script.

setup(
    name="wml_ai_model_managers",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author="Windmillcode",
    author_email="dev@windmillcode.com",
    license='MIT',
    packages=find_packages(),
    install_requires=[],
    keywords=['ai','ml','ml train','ml test','pytorch','ml text'],
    classifiers= [
        "Intended Audience :: Developers",
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3",
    ]
)
