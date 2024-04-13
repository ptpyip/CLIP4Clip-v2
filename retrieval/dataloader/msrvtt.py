from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import numpy as np
import torch
import pandas as pd

from .rawvideo_util import RawVideoExtractor
from .retrievalDataset import RetrievalDataset

class MSRVTTDataset(RetrievalDataset):
    """
    Dataset download: 
        - captions : https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip
        - video: https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip
    """
    DATASET_SPLIT = {
       "train": "MSRVTT_data.csv",
    #    "val": None, 
       "test": "MSRVTT_JSFUSION_test.csv",

    }

    def __init__(
        self,
        subset,
        data_dir,
        video_dir,
        tokenizer,
        max_words=30,
        framerate=1,
        max_frames=100,
        image_resolution=224,
    ):
        self.data_dir = data_dir
        self.video_dir = video_dir
        self.framerate = framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.rawVideoExtractor = RawVideoExtractor(framerate=framerate, size=image_resolution)

        self.subset = subset
        assert self.subset in self.DATASET_SPLIT.keys()

        data_path = os.path.join(self.data_dir, self.DATASET_SPLIT[subset])
        assert os.path.exists(data_path)
        
        self.data = pd.read_csv(data_path)
        if self.subset == "test":
            self.video_sentence_pairs = self.data[["video_id", "sentence"]].iloc
        else:
            self.video_sentence_pairs = self.data.iloc
            
        self.get_video_path = lambda video_id: os.path.join(self.video_dir, f"{video_id}.mp4")

        self.sample_len = len(self.data)


    def __len__(self):
        return self.sample_len
    
    
    def __getitem__(self, idx):
        video_id, sentence = self.video_sentence_pairs[idx]

        text = self._get_text(sentence)
        video, video_mask = self._get_rawvideo(video_id)
        return text, video, video_mask

    
    def _get_text(self, sentence):
        txt_tokens = self.tokenizer.tokenize(sentence)
        
        ## add special tokens w/ truncation
        if len(txt_tokens) >  self.max_words - 2:
            txt_tokens = txt_tokens[:self.max_words - 2]            
        txt_tokens = [self.SPECIAL_TOKEN["START_TOKEN"], *txt_tokens, self.SPECIAL_TOKEN["END_TOKEN"]]
        
        txt_token_ids = self.tokenizer.convert_tokens_to_ids(txt_tokens)
        if len(txt_token_ids) < self.max_words:
            paddings =  [0] * (self.max_words - len(txt_token_ids)) 
            txt_token_ids.extend(paddings)

        assert len(txt_token_ids) == self.max_words
        
        return torch.Tensor(txt_token_ids).to(torch.int64)        # [max_words]
    
    
    def _get_rawvideo(self, video_id):
        video_mask = np.zeros((self.max_frames), dtype=np.int64)
        video = np.zeros((
            self.max_frames, 3, self.rawVideoExtractor.size, self.rawVideoExtractor.size
        ), dtype=np.float64)  # 1 x L x 3 x H x W
        
        video_path = self.get_video_path(video_id)
        raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
        raw_video_data = raw_video_data['video']
        
        
        raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data)   # L x 3 x H x W
        sample_idx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
        raw_video_slice = raw_video_slice[sample_idx, ...]
            
        video_slice = self.rawVideoExtractor.process_frame_order(raw_video_slice)
        video_length = video_slice.shape[0]
        
        video[:video_length] = video_slice 
        video_mask[:video_length] = [1] * video_length 

        return video, video_mask
    