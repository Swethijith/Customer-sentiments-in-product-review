import os

def create_path(folder_path:str)->None:
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return None