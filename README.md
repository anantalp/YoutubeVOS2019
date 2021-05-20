## *Youtube Video Object Segmentation 2019*

### Introduction:
This was a course project for the class CAP6412 - Advanced Computer Vision, Spring 2019. It is a re-implementation of [Youtube-VOS](https://arxiv.org/pdf/1809.00461.pdf) by Xu et al.

### Abstract:
Getting to learn spatiotemporal features is critical for many video-related task analysis. Unfortunately, existing methodologies rely on static image
segmentation techniques and are required to depend upon pretrained optical flow models, leading to suboptimal solutions for the problem. End-to-end
sequential learning to explore spatiotemporal features for video segmentation is largely limited due to the constraint of the limited dataset. To mitigate
this problem, a new large-scale video object segmentation dataset called YouTube Video Object Segmentation dataset (YouTube-VOS) is used in this
project. It contains 3,252 YouTube video clips and 78 categories including common objects and human activities. The dataset used in this project Youtube-VOS dataset is by far the
largest. Using this dataset, a novel sequence-to-sequence network to fully exploit long-term spatial-temporal information in videos for
segmentation is implemented. We show that our method is able to achieve the best results on the YouTube-VOS test set and similar results on DAVIS 2016 compared to the current state-of-the-art
method. 

### Network Overview:
<img src="https://github.com/anantalp/YoutubeVOS2019/blob/main/figures/fig1.PNG">

### Dataset Description:
The dataset used for this project is [Youtube-VOS](https://youtube-vos.org/) dataset. Each instance level annotation along with the frame associated with it is mentioned in the meta.json file which will be available in the dataset file as this is an instance level segmentation task. While training, every 5th frame is annotated. That is, we have the segmentation ground truth for every 5th frame in the set. However, in the validation set, the first frame is annotated and our model is excepted to generate the instance segmentation for the corresponding object in consecutive frames.

### Network Training:
Use or modify the required dataloader present in the dataloader folder. To train the model, run Python train.py. For inference, use eval.py. The inference is mainly done by getting the segmentation mask (the probability scores of pixel values) for each object in a frame, and combining them to get instance level segmentation for the whole frame.

### Visualization:
<img src="https://github.com/anantalp/YoutubeVOS2019/blob/main/figures/fig2.png">
<img src="https://github.com/anantalp/YoutubeVOS2019/blob/main/figures/fig3.png">

Above are two qualitative results. The first roware the RGB frames, second is the ground truth segmentation masks, and last row is our  prediction.

### Results:
|Overall   |J_seen   |J_unseen   |F_seen   |F_unseen  |
|---|---|---|---|---|
|0.321|0.466|0.208|0.498|0.397| 


