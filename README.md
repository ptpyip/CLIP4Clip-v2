# CLIP4Clip: An Empirical Study of CLIP for End to End Video Clip Retrieval

Modified implementation of paper [**CLIP4Clip: An Empirical Study of CLIP for End to End Video Clip Retrieval**](https://arxiv.org/abs/2104.08860). 
(Official repo can be found [here](https://github.com/ArrowLuo/CLIP4Clip))

CLIP4Clip is a video-text retrieval model based on [CLIP (ViT-B)](https://github.com/openai/CLIP). We investigate three similarity calculation approaches: parameter-free type, sequential type, and tight type, in this work. The model achieve SOTA results on MSR-VTT, MSVD, LSMDC, ActivityNet, and DiDeMo.

![CLIP4Clip.png](https://github.com/ArrowLuo/CLIP4Clip/blob/master/CLIP4Clip.png)

## Requirement
```sh
# From CLIP
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm
pip install opencv-python boto3 requests pandas
```