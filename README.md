# CHOMP: Occluded Human Body Capture with Self-Supervised Spatial-Temporal Motion Prior

[Buzhen Huang](http://www.buzhenhuang.com/), Yuan Shu, Jingyi Ju, [Yangang Wang](https://www.yangangwang.com/)<br>
\[[Arxiv](https://arxiv.org/pdf/2207.05375.pdf)\]


![figure](/assets/pipline.jpg)

## Code is coming soon!


## OcMotion dataset

We build 3D Occluded Motion dataset (OcMotion) to reduce the gap between synthetic and real occlusion data, which contains 43 motions and 300K frames with accurate 3D annotations. OcMotion is the first video dataset explicitly designed for the occlusion problem.
![figure](/assets/dataset.jpg)

\[[Download Link](https://pan.baidu.com/s/14Yxz5mt9G-WeU8TK_Lg_yw?pwd=w3yc)\]

#### Visualize 2D keypoints and bounding-box:
```
python vis_OcMotion.py --dataset_dir PATH/TO/OcMotion  --output_dir output 
```
#### Visualize 3D meshes:
```
python vis_OcMotion.py --dataset_dir PATH/TO/OcMotion  --output_dir output --vis_smpl True
```

## Citation
If you use the data in your research, please consider citing at least one of the following references:
```
@article{huang2022object,
  title={Object-Occluded Human Shape and Pose Estimation with Probabilistic Latent Consistency},
  author={Huang, Buzhen and Zhang, Tianshu and Wang, Yangang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
  publisher={IEEE}
}
```
```
@article{huang2022occluded,
  title={Occluded Human Body Capture with Self-Supervised Spatial-Temporal Motion Prior},
  author={Huang, Buzhen and Shu, Yuan and Ju, Jingyi and Wang, Yangang},
  journal={arXiv preprint arXiv:2207.05375},
  year={2022}
}
```
