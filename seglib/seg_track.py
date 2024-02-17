import numpy as np
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from tqdm import tqdm
from emu.io import read_vol, compute_bbox_all
from emu.seg import seg_to_iou, seg_biggest_cc, seg_to_cc
from .region_graph import merge_id



def seg_to_global_id(filenames, index):
    num_vol = get_file_number(filenames, index)
    count = np.zeros(1 + num_vol, int)
    for i in range(num_vol):
        filename = get_filename(filenames, index, i)
        count[i + 1] = read_vol(filename).max()
    return np.cumsum(count)
    




def seg_to_iou(seg0, seg1, uid0=None, bb0=None, uid1=None, uc1=None):
    """
    Compute the intersection over union (IoU) between segments in two segmentation maps.

    Args:
        seg0 (numpy.ndarray): The first segmentation map.
        seg1 (numpy.ndarray): The second segmentation map.
        uid0 (numpy.ndarray, optional): The segment IDs to compute IoU for in the first segmentation map. Defaults to None.
        bb0 (numpy.ndarray, optional): The bounding boxes of segments in the first segmentation map. Defaults to None.
        uid1 (numpy.ndarray, optional): The segment IDs in the second segmentation map. Defaults to None.
        uic2 (numpy.ndarray, optional): The segment counts in the second segmentation map. Defaults to None.

    Returns:
        numpy.ndarray: An array containing the segment IDs, the best matching segment IDs, the segment counts, and the maximum overlap counts.

    Notes:
        - The function computes the intersection over union (IoU) between segments in two segmentation maps.
        - The IoU is computed for the specified segment IDs in `uid0`.
        - If `uid0` is not provided, the IoU is computed for all unique segment IDs in `seg0`.
    """    
    assert (np.array(seg0.shape) - seg1.shape).abs().max()==0, "seg0 and seg1 should have the same shape"
    if bb0 is not None:
        if seg0.ndim == 2:
            assert bb0.shape[1] == 6, "input bounding box for 2D segment has 6 columns [seg_id, ymin, ymax, xmin, xmax, count]"        
        elif seg0.ndim == 3:
            assert bb0.shape[1] == 8, "input bounding box for 3D segment has 8 columns [seg_id, zmin, zmax, ymin, ymax, xmin, xmax, count]"        
        else:
            raise "segment should be either 2D or 3D"                    
        
    # seg0 info: uid0, uc1, bb0
    # uid0 can be a subset of seg ids
    if uid0 is None:
        if bb0 is None:
            bb0 = compute_bbox_all(seg0, True)  
        uid0 = bb0[:, 0]
    elif bb0 is None:
        bb0 = compute_bbox_all(seg0, True, uid0)
    else:
        # select the boxes correspond to uid0
        bb0 = bb0[np.in1d(bb0[:, 0], uid0)]
        uid0 = bb0[:, 0]
    uc0 = bb0[:, -1]         

    # seg1 info: uid1, uc1
    if uid1 is None or uc1 is None:            
        uid1, uc1 = np.unique(seg1, return_counts=True)

    out = np.zeros((len(uid0), 5), int)
    out[:, 0] = uid0
    out[:, 2] = uc0

    for j, i in enumerate(uid0):
        bb = bb0[j, 1:]
        if seg0.ndim ==2:
            ui3, uc3 = np.unique(
                seg1[bb[0] : bb[1] + 1, bb[2] : bb[3] + 1]
                * (seg0[bb[0] : bb[1] + 1, bb[2] : bb[3] + 1] == i),
                return_counts=True,
            )
        else:
            ui3, uc3 = np.unique(
            seg1[bb[0] : bb[1] + 1, bb[2] : bb[3] + 1, bb[4] : bb[5] + 1]
            * (
                seg0[bb[0] : bb[1] + 1, bb[2] : bb[3] + 1, bb[4] : bb[5] + 1]
                == i
            ),
            return_counts=True,
        )
        uc3[ui3 == 0] = 0
        if (ui3 > 0).any():
            out[j, 1] = ui3[np.argmax(uc3)]
            out[j, 3] = uc1[uid1 == out[j, 1]]
            out[j, 4] = uc3.max()
    return out


def vol_to_iou(seg3d, th_iou=0):
    # raw iou result or matches
    ndim = 5 if th_iou == 0 else 2
    out = [np.zeros([ndim, 0])] * (seg3d.shape[0] - 1)
    bb_pre = compute_bbox_all(seg3d[0], True)
    for z in range(seg3d.shape[0] - 1):
        bb_new = compute_bbox_all(seg3d[z + 1], True)
        if bb_pre is not None and bb_new is not None:
            iou = seg_to_iou(seg3d[z], seg3d[z + 1], bb0=bb_pre, uid1=bb_new[:,0], uc1=bb_new[:,-1])
            if iou is not None:
                if th_iou == 0:
                    out[z] = iou.T
                else:
                    # remove matches to 0
                    iou = iou[iou[:, 1] != 0]
                    sc = iou[:, 4].astype(float) / (
                        iou[:, 2] + iou[:, 3] - iou[:, 4]
                    )
                    gid = sc > th_iou
                    out[z] = iou[gid, :2].T
        bb_pre = bb_new
    return np.hstack(out)


def iou_to_matches(fn_iou, im_id, global_id=None, th_iou=0.1):
    # assume each 2d seg id is not overlapped
    mm = [None] * (len(im_id))
    for z in tqdm(range(len(im_id))):
        iou = read_vol(fn_iou % im_id[z])
        sc = iou[:, 4].astype(float) / (iou[:, 2] + iou[:, 3] - iou[:, 4])
        gid = sc > th_iou
        mm[z] = iou[gid, :2].T
        if global_id is not None:
            mm[z][0] += global_id[z]
            mm[z][1] += global_id[z + 1]
    return np.hstack(mm)


def seg2d_mapping(seg, mapping):
    mapping_len = np.uint64(len(mapping))
    mapping_max = mapping.max()
    ind = seg < mapping_len
    seg[ind] = mapping[seg[ind]]  # if within mapping: relabel
    seg[np.logical_not(ind)] -= (
        mapping_len - mapping_max
    )  # if beyond mapping range, shift left
    return seg


def seg2d_to_global_id(seg, mapping=None, mid=None, th_sz=-1):
    if mapping is None:
        mid = mid.astype(np.uint32)
        mapping = merge_id(mid[0], mid[1])

    seg = seg2d_mapping(seg, mapping)
    if th_sz > 0:
        seg = remove_small_objects(seg, th_sz)
    return seg


def seg2d_to_3d(seg, matches=None, iou=None, th_iou=0.1, th_sz=-1):
    if matches is None:
        matches = iou_to_matches(iou, th_iou)
    seg = seg2d_to_global_id(seg, mid=matches, th_sz=th_sz)
    return seg
