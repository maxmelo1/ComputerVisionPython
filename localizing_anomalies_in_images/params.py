import torch


IMAGE_DIR = 'cell_images/'
IMAGE_SIZE = 224

LEARNING_RATE = 1e-3
BS = 32
NUM_WORKERS = 4
PIN_MEMORY = True
NUM_EPOCHS = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"