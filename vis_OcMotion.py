'''
 @FileName    : vis_dataset.py
 @EditTime    : 2022-07-10 16:40:05
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''

import os
import pickle as pkl
import torch
import cv2
import numpy as np
from utils.module_utils import *
from utils.smpl_torch_batch import SMPLModel
from utils.render import Renderer
import argparse

def vis_kpt_2d(dataset_dir='./3DMPB', output_dir='./output', vis_train=False, **kwargs):
    if vis_train:
        pkl_file = os.path.join(dataset_dir, 'annot', 'train.pkl')
    else:
        pkl_file = os.path.join(dataset_dir, 'annot', 'test.pkl')
    with open(pkl_file, 'rb') as f:
        annotations = pkl.load(f, encoding='latin1')
    
    colors = [[255, 0, 0], 
            [255, 255, 0],
            [0, 255, 0],
            [0, 255, 255], 
            [0, 0, 255], 
            [255, 0, 255]]

    for seq in annotations:
        for cam in seq:
            for frame in cam:
                img_path = frame['img_path']
                height, width = frame['h_w']
                img = cv2.imread(os.path.join(dataset_dir, img_path))

                for key in frame:
                    if key in ['h_w', 'img_path']:
                        continue
                    bbox_tmp = frame[key]['bbox']
                    bbox = np.array(bbox_tmp).astype(np.int32).reshape(-1)  # 2*2->4*1
                    img = draw_bbox(img, bbox, thickness=3, color=colors[int(key)%len(colors)])
                    kpt = frame[key]['lsp_joints_2d'][:,:2]
                    img = draw_skeleton(img, kpt, connection=None, colors=colors[int(key)%len(colors)], bbox=bbox)
                    
                vis_img('img', img)
                output_path = os.path.join(output_dir, 'vis_2d', img_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, img)
    print('Finish!')


def vis_smpl_3d(dataset_dir='./3DMPB', output_dir='./output', vis_train=False, **kwargs):
    if vis_train:
        pkl_file = os.path.join(dataset_dir, 'annot', 'train.pkl')
    else:
        pkl_file = os.path.join(dataset_dir, 'annot', 'test.pkl')
    with open(pkl_file, 'rb') as f:
        annotations = pkl.load(f, encoding='latin1')

    smpl = SMPLModel(device=torch.device('cpu'), model_path='data/SMPL_NEUTRAL.pkl')
    render = Renderer(resolution=(annotations[0][0][0]['h_w'][1], annotations[0][0][0]['h_w'][0]))

    for seq in annotations:
        for cam in seq:
            for f, frame in enumerate(cam):
                img_path = frame['img_path']
                height, width = frame['h_w']
                img = cv2.imread(os.path.join(dataset_dir, img_path))

                pose, shape, trans = [], [], []
                for key in frame:
                    if key in ['h_w', 'img_path']:
                        continue
                    pose.append(frame[key]['pose'])
                    shape.append(frame[key]['betas'])
                    trans.append(frame[key]['trans'])

                    intri = np.array(frame[key]['intri'], dtype=np.float32)
                    extri = np.array(frame[key]['extri'], dtype=np.float32)

                pose = torch.from_numpy(np.array(pose, dtype=np.float32).reshape(-1, 72))
                shape = torch.from_numpy(np.array(shape, dtype=np.float32).reshape(-1, 10))
                trans = torch.from_numpy(np.array(trans, dtype=np.float32).reshape(-1, 3))

                verts, joints = smpl(shape, pose, trans)

                img = render.render_multiperson(verts.detach().cpu().numpy(), smpl.faces, extri[:3,:3], extri[:3,3], intri, img.copy(), viz=False)

                vis_img('img', img)
                output_path = os.path.join(output_dir, 'vis_3d', img_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, img)


def main(vis_smpl=False, **kwargs):
    if vis_smpl:
        vis_smpl_3d(**kwargs)
    else:
        vis_kpt_2d(**kwargs)
    

if __name__ == "__main__":
    # sys.argv = ['', '--dataset_dir=//105.1.1.110/e/dataset_processed/OCMotion-public', '--output_dir=output', '--vis_train=True', '--vis_smpl=False']

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='directory of dataset')
    parser.add_argument('--vis_smpl', default=False, type=bool, help='')
    parser.add_argument('--vis_train', default=False, type=bool, help='')
    parser.add_argument('--output_dir', default='./output', type=str, help='directory of output images')
    args = parser.parse_args()
    args_dict = vars(args)

    main(**args_dict)
    

    