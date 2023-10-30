import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse

from wml_ai_model_managers.text_model_manager_zero.model_manager import WMLTextModelManagerZero




myai = WMLTextModelManagerZero()
myai.load_model_via_pytorch()
myai.chat_with_model()



