import os


def xor(a, b):
    return bool(a) != bool(b)

def find_file(file_name, search_path="."):
    result = []

    for root, dirs, files in os.walk(search_path):
        if file_name in files:
            result.append(os.path.join(root, file_name))

    return result
