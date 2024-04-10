from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import pickle
from .rawvideo_util import RawVideoExtractor

from .retrievalDataset import RetrievalDataset

class MSVD_Dataset(Dataset):
    DATASET_SPLIT = {
       "train": "train_list.txt",
       "val": "val_list.txt",
       "test": "test.txt",

    }
    SPECIAL_TOKEN = {
        # "CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>"
        "START_TOKEN": "<|startoftext|>", "END_TOKEN": "<|endoftext|>",
        "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"
    }

    def __init__(
            self,
            subset,
            data_dir,
            video_dir,
            tokenizer,
            max_words=30,
            framerate=1.0,
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

        video_id_path = os.path.join(self.data_dir, self.DATASET_SPLIT[subset])
        with open(video_id_path, 'r') as fp:
            video_ids = [itm.strip() for itm in fp.readlines()]
        
        caption_file = os.path.join(self.data_dir, "raw-captions.pkl")
        with open(caption_file, 'rb') as f:
            captions = pickle.load(f)

        self.video_paths: dict = self._parse_video_paths(video_dir, video_ids)

        self.video_sentence_pairs, self.cut_off_points = self._get_video_sentence_pairs(
           video_dir, video_ids 
        )

        ## below variables are used to multi-sentences retrieval
        # self.cut_off_points: used to tag the label when calculate the metric
        # self.sentence_num: used to cut the sentence representation
        # self.video_num: used to cut the video representation
        self.multi_sentence_per_video = True    # !!! important tag for eval
        if self.subset == "val" or self.subset == "test":
            self.sentence_num = len(self.video_sentence_pairs)
            self.video_num = len(video_ids)
            assert len(self.cut_off_points) == self.video_num
            print("For {}, sentence number: {}".format(self.subset, self.sentence_num))
            print("For {}, video number: {}".format(self.subset, self.video_num))

        print("Video number: {}".format(len(self.video_sentence_pairs)))
        print("Total Paire: {}".format(len(self.video_sentence_pairs)))

        self.sample_len = len(self.video_sentence_pairs)

    def __len__(self):
        return self.sample_len
    
    def __getitem__(self, idx):
        video_id, sentence = self.video_sentence_pairs[idx]

        pairs_text = self._get_text(sentence)
        video, video_mask = self._get_rawvideo(video_id)
        return pairs_text, video, video_mask
    
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
        
        return [np.array(txt_token_ids)]        # [1, max_words]
    
    def _get_rawvideo(self, video_id):
        video_length = [0] 
        video_mask = np.zeros((1, self.max_frames), dtype=np.int64)
        video = np.zeros((
            1, self.max_frames, 3, self.rawVideoExtractor.size, self.rawVideoExtractor.size
        ), dtype=np.float64)  # 1 x L x 3 x H x W
        
        video_path = self.video_paths[video_id]
        raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
        raw_video_data = raw_video_data['video']
        
        
        raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data)   # L x 3 x H x W
        sample_idx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
        raw_video_slice = raw_video_slice[sample_idx, ...]
            
        video_slice = self.rawVideoExtractor.process_frame_order(raw_video_slice)
        video_length = video_slice.shape[0]
        
        video[0][:video_length] = video_slice 
        video_mask[0][:video_length] = [1] * video_length 

        return video, video_mask
    
    @staticmethod
    def _parse_video_paths(video_dir, video_ids) -> dict:
        video_path_dict = {}
        for root, dub_dir, video_files in os.walk(video_dir):
            for video_file in video_files:
                video_id_ = ".".join(video_file.split(".")[:-1])
                if video_id_ not in video_ids:
                    continue
                file_path_ = os.path.join(root, video_file)
                video_path_dict[video_id_] = file_path_
        
        return video_path_dict

    @staticmethod
    def _get_video_sentence_pairs(video_ids, captions):
        video_sentence_pairs = []
        cut_off_points = []
        for video_id in video_ids:
            assert video_id in captions
            for cap in captions[video_id]:
                cap_txt = " ".join(cap)
                video_sentence_pairs.append((video_id, cap_txt))
            cut_off_points.append(len(video_sentence_pairs))

        return video_sentence_pairs, cut_off_points