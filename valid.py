from utils import *
import torch

model = torch.load('model/model.pt', map_location='cpu')
