import os

import torch


def xor(a, b):
    return bool(a) != bool(b)

def find_file(file_name, search_path="."):
    result = []

    for root, dirs, files in os.walk(search_path):
        if file_name in files:
            result.append(os.path.join(root, file_name))

    return result

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def create_and_write_to_file(file_path, content):
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_path, 'w',encoding="utf-8") as file:
        file.write(content)
