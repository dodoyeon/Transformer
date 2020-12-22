import torch
import numpy as np
from kmeans_pytorch import kmeans

class cluster_module():
    def __init__(self):
        super().__init__()
        self.kmean()