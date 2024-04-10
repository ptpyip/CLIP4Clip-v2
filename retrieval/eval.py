import numpy as np
from functools import partial
from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader

from ..clip4clip import CLIP4Clip
from .utils import metrics
from .utils.logging import logger
from .utils.distributed import parallel_apply
from .config import DataConfig
from .dataloader import RetrievalDataLoader, init_dataloader

# EvalEpochCallable = Callable[
#     [CLIP4Clip, torch.device, int, Optional[bool]], tuple[dict,dict]
# ]

def __eval_epoch(model, dataloader, device, n_gpu):  
    assert hasattr(dataloader.dataset, 'multi_sentence_query')  
    if dataloader.dataset.multi_sentence_query:
        return multi_sentence_eval_epoch(model, dataloader, device, n_gpu)
    else:
        return single_sentence_eval_epoch(model, dataloader, device, n_gpu)
    
def get_eval_epoch(config: DataConfig, local_rank):
    dataloader, sample_size, _ = init_dataloader(config, mode="test")
    
    if local_rank == 0:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", sample_size)
        logger.info("  Batch size = %d", config.eval_batch_size)
        logger.info("  Num steps = %d", len(dataloader))
        
    assert hasattr(dataloader.dataset, 'multi_sentence_query')  
    if dataloader.dataset.multi_sentence_query:
        return partial(multi_sentence_eval_epoch, dataloader=dataloader)
        # return lambda model, device, n_gpu: multi_sentence_eval_epoch(model, dataloader, device, n_gpu)
    else:
        return partial(single_sentence_eval_epoch, dataloader=dataloader)
        # return lambda model, device, n_gpu: single_sentence_eval_epoch(model, dataloader, device, n_gpu)      
        
def single_sentence_eval_epoch(
    model: CLIP4Clip, dataloader: DataLoader, 
    device, n_gpu, video_to_text_eval=False
):
    model = model.to(device) 
    model.eval()
    
    with torch.no_grad():
        batch_sequence_outputs, batch_visual_outputs = [], []
        
        ## 1. cache the features
        for bid, batch in enumerate(dataloader):
            text, video, video_mask = batch

            sequence_output = model.forward_text(text.to(device))
            batch_sequence_outputs.append(sequence_output)
            
            visual_output = model.forward_visual(video.to(device), video_mask.to(device))
            batch_visual_outputs.append(visual_output)

            print("{}/{}\r".format(bid, len(dataloader)), end="")

        ## 2. compute similarity 
        sim_matrix = compute_similarity(model, batch_sequence_outputs, batch_visual_outputs, n_gpu)
        logger.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        
        tv_metrics = metrics.compute_metrics(sim_matrix)
        vt_metrics = metrics.compute_metrics(sim_matrix.T) if video_to_text_eval else {}
        logger.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))
       
    logger.info("Text-to-Video:")
    logger.info(
        '\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
            format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR'])
    )
    
    if video_to_text_eval:
        logger.info("Video-to-Text:")
        logger.info('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                    format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))

    return tv_metrics, vt_metrics 

def multi_sentence_eval_epoch(model: CLIP4Clip, dataloader: RetrievalDataLoader, device, n_gpu, video_to_text_eval=False):
    model = model.to(device) 
    assert dataloader.dataset.multi_sentence_query 
    
    cut_off_points_ = dataloader.dataset.cut_off_points         # used to tag the label when calculate the metric
    sentence_num_ = dataloader.dataset.sentence_num             # cut the sentence representation
    video_num_ = dataloader.dataset.video_num                   # cut the video representation
    cut_off_points_ = [itm - 1 for itm in cut_off_points_]

    logger.warning("Eval under the multi-sentence per video clip setting.")
    logger.warning("sentence num: {}, video num: {}".format(sentence_num_, video_num_))
    
    model.eval()
    with torch.no_grad():
        batch_sequence_outputs, batch_visual_outputs = [], []
        total_video_num = 0
        
        for bid, batch in enumerate(dataloader):
            batch = tuple(t.to(device) for t in batch)
            text, video, video_mask = batch

            # multi-sentences retrieval means: one clip has two or more descriptions.
            b, *_t = video.shape
            sequence_output = model.forward_text(text)
            batch_sequence_outputs.append(sequence_output)

            s_, e_ = total_video_num, total_video_num + b
            filter_inds = [itm - s_ for itm in cut_off_points_ if itm >= s_ and itm < e_]

            if len(filter_inds) > 0:
                video, video_mask = video[filter_inds, ...], video_mask[filter_inds, ...]
                visual_output = model.forward_visual(video, video_mask)
                batch_visual_outputs.append(visual_output)
            total_video_num += b

            print("{}/{}\r".format(bid, len(dataloader)), end="")

        sim_matrix = compute_similarity(model, batch_sequence_outputs, batch_visual_outputs, n_gpu)
        logger.info("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        max_length = max([e_-s_ for s_, e_ in zip([0]+cut_off_points2len_[:-1], cut_off_points2len_)])
        sim_matrix_new = []
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            sim_matrix_new.append(np.concatenate(
                (sim_matrix[s_:e_], np.full((max_length-e_+s_, sim_matrix.shape[1]), -np.inf)
            ), axis=0))
        
        sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
        logger.info(
            "after reshape, sim matrix size: {} x {} x {}".
                format(sim_matrix.shape[0], sim_matrix.shape[1], sim_matrix.shape[2])
        )

        tv_metrics = metrics.tensor_text_to_video_metrics(sim_matrix)
        vt_metrics = metrics.compute_metrics(tensor_video_to_text_sim(sim_matrix))  if video_to_text_eval else {}
        logger.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))
        
    logger.info("Text-to-Video:")
    logger.info(
        '\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
            format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR'])
    )
    
    if video_to_text_eval:
        logger.info("Video-to-Text:")
        logger.info('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                    format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))

    return tv_metrics, vt_metrics 
    

def compute_similarity(model, batch_sequence_outputs, batch_visual_outputs, n_gpu):
    if n_gpu <= 1:
        sim_matrix = _compute_similarity_on_single_gpu(
            model, batch_sequence_outputs, batch_visual_outputs
        )
        sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
        return sim_matrix
    
    ## else n_gpu > 1:
    device_ids = list(range(n_gpu))
    bacth_len = len(batch_sequence_outputs)
    split_len = (bacth_len + n_gpu - 1) // n_gpu
    
    parameters_tuple_list = []
    for dev_id in device_ids:
        s_, e_ = dev_id * split_len, (dev_id + 1) * split_len
        
        if dev_id == 0:
            batch_t_output_split = batch_sequence_outputs[s_:e_]
            batch_v_output_split = batch_visual_outputs
        else:
            device = torch.device('cuda:{}'.format(str(dev_id)))
            batch_t_output_split = [b.to(device) for b in batch_sequence_outputs[s_:e_]]
            
            batch_v_output_split = [b.to(device) for b in batch_visual_outputs]

        parameters_tuple_list.append(
            (batch_t_output_split, batch_v_output_split)
        )   
        
    parallel_outputs = parallel_apply(
        _compute_similarity_on_single_gpu, model, parameters_tuple_list, device_ids
    )
    
    ### gather
    sim_matrix = []
    for idx in range(len(parallel_outputs)):
        sim_matrix += parallel_outputs[idx]
        
    sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
    return sim_matrix

## TODO: 
def _compute_similarity_on_single_gpu(
    model, batch_sequence_outputs, batch_visual_outputs
):
    sim_matrix = []
    for sequence_output in batch_sequence_outputs:
        row = []
        for visual_output in batch_visual_outputs:
            logits = model.get_similarity_logits(sequence_output, visual_output)
            row.append(logits.cpu().detach().numpy())

        sim_matrix.append(np.concatenate(tuple(row), axis=-1))
        
    return sim_matrix

def tensor_video_to_text_sim(sim_tensor):
    if not torch.is_tensor(sim_tensor):
      sim_tensor = torch.tensor(sim_tensor)
    # Code to avoid nans
    sim_tensor[sim_tensor != sim_tensor] = float('-inf')
    # Forms a similarity matrix for use with rank at k
    values, _ = torch.max(sim_tensor, dim=1, keepdim=True)
    return torch.squeeze(values).T
