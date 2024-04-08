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
        "CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
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

        self.video_paths: dict = self.parse_video_paths(video_dir, video_ids)

        self.video_sentence_pairs, self.cut_off_points = self.get_video_sentence_pairs(
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
    
    @staticmethod
    def parse_video_paths(video_dir, video_ids) -> dict:
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
    def get_video_sentence_pairs(video_ids, captions):
        video_sentence_pairs = []
        cut_off_points = []
        for video_id in video_ids:
            assert video_id in captions
            for cap in captions[video_id]:
                cap_txt = " ".join(cap)
                video_sentence_pairs.append((video_id, cap_txt))
            cut_off_points.append(len(video_sentence_pairs))

        return video_sentence_pairs, cut_off_points