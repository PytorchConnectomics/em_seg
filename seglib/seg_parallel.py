import os 
import numpy as np
from emu.io import compute_bbox_all
from . import seg_to_iou

def vol_to_iou_parallel(get_seg, filename, index, th_iou=0):
    # compute consecutive iou among the indices
    # raw iou result or matches
    if not os.path.exists(filename): 
        seg0 = get_seg(index[0])
        bb0 = compute_bbox_all(seg0, True)
        out =[[]] * (len(index) - 1)
        for i, z in enumerate(index[1:]):
            seg1 = get_seg(z)
            bb1 = compute_bbox_all(seg1, True)    
            iou = seg_to_iou(seg0, seg1, bb0=bb0, uid1=bb1[:, 0], uc1=bb1[:, -1])    
            if th_iou == 0:
                # store all iou
                out[i] = iou.T
            else:
                # store matches
                # remove background seg id            
                iou = iou[iou[:, 1] != 0]
                score = iou[:, 4].astype(float) / (
                    iou[:, 2] + iou[:, 3] - iou[:, 4]
                )
                gid = score > th_iou
                out[i] = iou[gid, :2].T
            bb0 = bb1            
        np.save(filename, out)
