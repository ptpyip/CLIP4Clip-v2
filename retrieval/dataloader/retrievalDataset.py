import torch
from torch.utils.data import Dataset

class RetrievalDataset(Dataset):
    multi_sentence_query: bool = False
    
    cut_off_points: list        # used to tag the label when calculate the metric
    sentence_num: int           # cut the sentence representation
    video_num: int              # cut the video representation
    
    def __getitem__(self, idx):
        raise NotImplementedError
