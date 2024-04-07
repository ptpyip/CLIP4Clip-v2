import torch
from torch.utils.data import DataLoader

from retrievalDataset import RetrievalDataset

class RetrievalDataLoader(DataLoader):
    dataset: RetrievalDataset