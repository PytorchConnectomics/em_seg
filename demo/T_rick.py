import numpy as np
from emu.io import read_vol,write_h5, seg_relabel, seg_postprocess
from emu.eval import adapted_rand
from seg_volume import affToSeg2d,seg2dToseg3d,mergeWithSize 

if __name__ == "__main__":
    aff_uint8 = read_vol('aff_xy.h5')
    wz_thres = 0.7

    # compute seg
    seg2d = affToSeg2d(aff_uint8)
    seg3d = seg2dToseg3d(aff_uint8, seg2d, wz_thres)
    for i,thd in enumerate(wz_thres):
        print('process thd=%.3f'%thd)
        seg3d[i] = seg3d[i].astype(np.uint64)
        # merge small seg
        mergeWithSize(aff_uint8, seg3d[i])
        # post-processing
        seg3d[i] = seg_postprocess(seg_relabel(seg3d[i], True))

        # eval
        segEval(seg3d[i][:,::ratio,::ratio], test_gt)
    write_h5('output', seg3d)