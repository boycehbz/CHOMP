'''
 @FileName    : module_utils.py
 @EditTime    : 2023-02-10 19:38:40
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''

import cv2
import math
import numpy as np


def vis_img(name, im):
    ratiox = 800/int(im.shape[0])
    ratioy = 800/int(im.shape[1])
    if ratiox < ratioy:
        ratio = ratiox
    else:
        ratio = ratioy

    cv2.namedWindow(name,0)
    cv2.resizeWindow(name,int(im.shape[1]*ratio),int(im.shape[0]*ratio))
    #cv2.moveWindow(name,0,0)
    if im.max() > 1:
        im = im/255.
    cv2.imshow(name,im)
    if name != 'mask':
        cv2.waitKey()

def draw_bbox(img, bbox, thickness=3, color=(255, 0, 0)):
    canvas = img.copy()
    cv2.rectangle(canvas, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
    return canvas

def draw_skeleton(img, kpt, connection=None, colors=None, bbox=None):
    kpt = np.array(kpt)
    npart = kpt.shape[0]
    canvas = img.copy()
        
    if npart==14:
        part_names = ['Left_Hip', 'Right_Hip', 'Waist', 'Left_Knee', 'Right_Knee',
                      'Upper_Waist', 'Left_Ankle', 'Right_Ankle', 'Chest',
                      'Left_Toe', 'Right_Toe', 'Base_Neck', 'Left_Shoulder',
                      'Right_Shoulder', 'Upper_Neck', 'Left_Arm', 'Right_Arm',
                      'Left_Elbow', 'Right_Elbow', 'Left_Wrist', 'Right_Wrist',
                      'Left_Finger', 'Right_Finger']
        visible_map = {0: 'missing', 
                       1: 'vis'}
        map_visible = {value: key for key, value in visible_map.items()}
        if connection is None:
            connection = [[13, 12], [12, 8], [8, 7], [7, 6], 
                        [12, 9], [9, 10], [10, 11], 
                        [12, 2], [2, 1], [1, 0], 
                        [12, 3], [3, 4], [4, 5]]                      
        idxs_draw = [i for i in range(kpt.shape[0])]
                
    if colors is None:
        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], 
                 [255, 255, 0], [170, 255, 0], [85, 255, 0], 
                 [0, 255, 0], [0, 255, 85], [0, 255, 170], 
                 [0, 255, 255], [0, 170, 255], [0, 85, 255], 
                 [0, 0, 255], [85, 0, 255], [170, 0, 255],
                 [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    elif type(colors[0]) not in [list, tuple]:
        colors = [colors]
    
    if len(idxs_draw)==0:
        return img
    
    if bbox is None:
        bbox = [np.min(kpt[idxs_draw, 0]), np.min(kpt[idxs_draw, 1]),
                np.max(kpt[idxs_draw, 0]), np.max(kpt[idxs_draw, 1])] # xyxy
    
    Rfactor = math.sqrt((bbox[2]-bbox[0]) * (bbox[3]-bbox[1])) / math.sqrt(img.shape[0] * img.shape[1])
    Rpoint = int(min(10, max(Rfactor*10, 4)))
    Rline = int(min(10, max(Rfactor*5, 2)))
    
    for idx in idxs_draw:
        if kpt.shape[1] == 2:
            x, y = kpt[idx, :]
            cv2.circle(canvas, (x, y), Rpoint, colors[idx%len(colors)], thickness=-1)
        else:
            x, y, v = kpt[idx, :]
            cv2.circle(canvas, (x, y), Rpoint, colors[idx%len(colors)], thickness=-1)
            
            if v==2:
                cv2.rectangle(canvas, (x-Rpoint-1, y-Rpoint-1), (x+Rpoint+1, y+Rpoint+1), 
                            colors[idx%len(colors)], 1)
            elif v==3:
                cv2.circle(canvas, (x, y), Rpoint+2, colors[idx%len(colors)], thickness=1)

    for idx in range(len(connection)):
        idx1, idx2 = connection[idx]
        if kpt.shape[1] == 2:
            y1, x1 = kpt[idx1]
            y2, x2 = kpt[idx2]
            v1 = 1
            v2 = 1
        else:
            y1, x1, v1 = kpt[idx1-1]
            y2, x2, v2 = kpt[idx2-1]
        if v1 == map_visible['missing'] or v2 == map_visible['missing']:
            continue
        mX = (x1+x2)/2.0
        mY = (y1+y2)/2.0
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        angle = math.degrees(math.atan2(x1 - x2, y1 - y2))
        polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), Rline), int(angle), 0, 360, 1)
        cur_canvas = canvas.copy()
        cv2.fillConvexPoly(cur_canvas, polygon, colors[idx%len(colors)])
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        
    return canvas

def draw_mask(img, mask, thickness=3, color=(255, 0, 0)):
    def _get_edge(mask, thickness=3):
        dtype = mask.dtype
        x=cv2.Sobel(np.float32(mask),cv2.CV_16S,1,0, ksize=thickness) 
        y=cv2.Sobel(np.float32(mask),cv2.CV_16S,0,1, ksize=thickness)
        absX=cv2.convertScaleAbs(x)
        absY=cv2.convertScaleAbs(y)  
        edge = cv2.addWeighted(absX,0.5,absY,0.5,0)
        return edge.astype(dtype)
    
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    img = img.copy()
    canvas = np.zeros(img.shape, img.dtype) + color
    img[mask > 0] = img[mask > 0] * 0.8 + canvas[mask > 0] * 0.2
    edge = _get_edge(mask, thickness)
    img[edge > 0] = img[edge > 0] * 0.2 + canvas[edge > 0] * 0.8
    return img

def surface_projection(vertices, faces, joint, extri, intri, image, viz=False):
    """
    @ vertices: N*3, mesh vertex
    @ faces: N*3, mesh face
    @ joint: N*3, joints
    @ extri: 4*4, camera extrinsic
    @ intri: 3*3, camera intrinsic
    @ image: RGB image
    @ viz: bool, visualization
    """
    im = np.ascontiguousarray(image.copy(), dtype=np.uint8)
    h = im.shape[0]
    # homogeneous
    intri_ = np.insert(intri,3,values=0.,axis=1)
    temp_v = np.insert(vertices,3,values=1.,axis=1).transpose((1,0))

    # projection
    out_point = np.dot(extri, temp_v)
    dis = out_point[2]
    out_point = (np.dot(intri_, out_point) / dis)[:-1]
    out_point = (out_point.astype(np.int32)).transpose(1,0)
    
    # color
    max = dis.max()
    min = dis.min()
    t = 255./(max-min)
    color = (255, 255, 255)
    
    # draw mesh
    for f in faces:
        point = out_point[f]
        im = cv2.polylines(im, [point], True, color, 1)
    
    # joints projection
    temp_joint = np.insert(joint,3,values=1.,axis=1).transpose((1,0))
    out_point = np.dot(extri, temp_joint)
    dis = out_point[2]
    out_point = (np.dot(intri_, out_point) / dis)[:-1].astype(np.int32)
    out_point = out_point.transpose(1,0)

    # # draw projected joints
    # for i in range(len(out_point)):
    #     im = cv2.circle(im, tuple(out_point[i]), int(h/100), (255,0,0),-1)

    # visualization
    if viz:
        ratiox = 800/int(im.shape[0])
        ratioy = 800/int(im.shape[1])
        if ratiox < ratioy:
            ratio = ratiox
        else:
            ratio = ratioy

        cv2.namedWindow("mesh",0)
        cv2.resizeWindow("mesh",int(im.shape[1]*ratio),int(im.shape[0]*ratio))
        cv2.moveWindow("mesh",0,0)
        cv2.imshow('mesh',im/255.)
        cv2.waitKey()

    return out_point, im
