import torch
from torch.utils.data import DataLoader

import numpy as np

from ..clip4clip import CLIP4Clip
from .utils.logging import logger
from .utils.distributed import parallel_apply
from .dataloader import RetrievalDataLoader

        
def single_sentence_eval_epoch(model: CLIP4Clip, dataloader: DataLoader, device, n_gpu):
    model = model.to(device) 
    model.eval()
    
    with torch.no_grad():
        batch_list_t = []
        batch_list_v = []
        batch_sequence_output_list, batch_visual_output_list = [], []
        total_video_num = 0
        
        # ----------------------------
        # 1. cache the features
        # ----------------------------
        for bid, batch in enumerate(dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, video, video_mask = batch
            
            sequence_output, visual_output = model.get_sequence_visual_output(input_ids, segment_ids, input_mask, video, video_mask)

            batch_sequence_output_list.append(sequence_output)
            batch_list_t.append((input_mask, segment_ids,))

            batch_visual_output_list.append(visual_output)
            batch_list_v.append((video_mask,))

            print("{}/{}\r".format(bid, len(dataloader)), end="")

        sim_matrix = compute_similarity(n_gpu)
        logger.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        
        tv_metrics = compute_metrics(sim_matrix)
        vt_metrics = compute_metrics(sim_matrix.T)
        logger.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))
        
    return tv_metrics, vt_metrics 

def multi_sentence_eval_epoch(model: CLIP4Clip, dataloader: RetrievalDataLoader, device, n_gpu):
    model = model.to(device) 
    
    assert hasattr(dataloader.dataset, 'multi_sentence_query') 
    assert dataloader.dataset.multi_sentence_query 
    
    cut_off_points_ = dataloader.dataset.cut_off_points         # used to tag the label when calculate the metric
    sentence_num_ = dataloader.dataset.sentence_num             # cut the sentence representation
    video_num_ = dataloader.dataset.video_num                   # cut the video representation
    cut_off_points_ = [itm - 1 for itm in cut_off_points_]

    logger.warning("Eval under the multi-sentence per video clip setting.")
    logger.warning("sentence num: {}, video num: {}".format(sentence_num_, video_num_))
    
    model.eval()
    with torch.no_grad():
        batch_list_t = []
        batch_list_v = []
        batch_sequence_output_list, batch_visual_output_list = [], []
        total_video_num = 0
        
        # ----------------------------
        # 1. cache the features
        # ----------------------------
        for bid, batch in enumerate(dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, video, video_mask = batch

            # multi-sentences retrieval means: one clip has two or more descriptions.
            b, *_t = video.shape
            sequence_output = model.get_sequence_output(input_ids, segment_ids, input_mask)
            batch_sequence_output_list.append(sequence_output)
            batch_list_t.append((input_mask, segment_ids,))

            s_, e_ = total_video_num, total_video_num + b
            filter_inds = [itm - s_ for itm in cut_off_points_ if itm >= s_ and itm < e_]

            if len(filter_inds) > 0:
                video, video_mask = video[filter_inds, ...], video_mask[filter_inds, ...]
                visual_output = model.get_visual_output(video, video_mask)
                batch_visual_output_list.append(visual_output)
                batch_list_v.append((video_mask,))
            total_video_num += b

            print("{}/{}\r".format(bid, len(dataloader)), end="")

        sim_matrix = compute_similarity(n_gpu)
        logger.info("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        max_length = max([e_-s_ for s_, e_ in zip([0]+cut_off_points2len_[:-1], cut_off_points2len_)])
        sim_matrix_new = []
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            sim_matrix_new.append(np.concatenate((sim_matrix[s_:e_],
                                                  np.full((max_length-e_+s_, sim_matrix.shape[1]), -np.inf)), axis=0))
        
        sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
        logger.info("after reshape, sim matrix size: {} x {} x {}".
                    format(sim_matrix.shape[0], sim_matrix.shape[1], sim_matrix.shape[2]))

        tv_metrics = tensor_text_to_video_metrics(sim_matrix)
        vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix)
        logger.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0]))))
        
    return tv_metrics, vt_metrics 
    

def compute_similarity(n_gpu):
    if n_gpu <= 1:
        sim_matrix = _compute_similarity_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list)
        sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
        return sim_matrix
    
    ## else n_gpu > 1:
    device_ids = list(range(n_gpu))
    bacth_len = len(batch_list_t)
    split_len = (bacth_len + n_gpu - 1) // n_gpu
    
    batch_list_t_splits = []
    batch_list_v_splits = []
    batch_t_output_splits = []
    batch_v_output_splits = []
    for dev_id in device_ids:
        s_, e_ = dev_id * split_len, (dev_id + 1) * split_len
        if dev_id == 0:
            batch_list_t_splits.append(batch_list_t[s_:e_])
            batch_list_v_splits.append(batch_list_v)

            batch_t_output_splits.append(batch_sequence_output_list[s_:e_])
            batch_v_output_splits.append(batch_visual_output_list)
        else:
            devc = torch.device('cuda:{}'.format(str(dev_id)))
            devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_t[s_:e_]]
            batch_list_t_splits.append(devc_batch_list)
            devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_v]
            batch_list_v_splits.append(devc_batch_list)

            devc_batch_list = [b.to(devc) for b in batch_sequence_output_list[s_:e_]]
            batch_t_output_splits.append(devc_batch_list)
            devc_batch_list = [b.to(devc) for b in batch_visual_output_list]
            batch_v_output_splits.append(devc_batch_list)

    parameters_tuple_list = [(batch_list_t_splits[dev_id], batch_list_v_splits[dev_id],
                                batch_t_output_splits[dev_id], batch_v_output_splits[dev_id]) for dev_id in device_ids]
    parallel_outputs = parallel_apply(_compute_similarity_on_single_gpu, model, parameters_tuple_list, device_ids)
    sim_matrix = []
    for idx in range(len(parallel_outputs)):
        sim_matrix += parallel_outputs[idx]
        
    sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
    return sim_matrix

## TODO: 
def _compute_similarity_on_single_gpu(
    model, 
    batch_list_t, batch_list_v, 
    batch_sequence_output_list, batch_visual_output_list
):
    sim_matrix = []
    for idx1, b1 in enumerate(batch_list_t):
        input_mask, segment_ids, *_tmp = b1
        sequence_output = batch_sequence_output_list[idx1]
        each_row = []
        for idx2, b2 in enumerate(batch_list_v):
            video_mask, *_tmp = b2
            visual_output = batch_visual_output_list[idx2]
            b1b2_logits, *_tmp = model.get_similarity_logits(
                sequence_output, visual_output, 
                input_mask, video_mask, 
                loose_type=model.loose_type
            )
            b1b2_logits = b1b2_logits.cpu().detach().numpy()
            each_row.append(b1b2_logits)
        each_row = np.concatenate(tuple(each_row), axis=-1)
        sim_matrix.append(each_row)
    return sim_matrix
