import numpy as np
from imu.io import readH5,writeH5
from imu.seg import adapted_rand,relabel,seg_postprocess
from seg_volume import affToSeg2d,seg2dToseg3d,mergeWithSize 

if __name__ == "__main__":
    # load gt
    test_gt = readH5('test-labels.h5')
    # load input affinity 
    aff_uint8 = readH5('aff_xy.h5')
    ratio = 2
    wz_thres = [0.7]

    # compute seg
    seg2d = affToSeg2d(aff_uint8)
    seg3d = seg2dToseg3d(aff_uint8, seg2d, wz_thres)
    for i,thd in enumerate(wz_thres):
        print('process thd=%.3f'%thd)
        seg3d[i] = seg3d[i].astype(np.uint64)
        # merge small seg
        mergeWithSize(aff_uint8, seg3d[i])
        # post-processing
        seg3d[i] = seg_postprocess(relabel(seg3d[i], True))

        # eval
        segEval(seg3d[i][:,::ratio,::ratio], test_gt)
