import numpy as np
import waterz
import zwatershed
from T_util import readh5,writeh5,adapted_rand,relabel,seg_postprocess

def affToSeg2d(aff_uint8, T_thres = 800, T_dust = 50, T_dust_merge = 0.2,T_mst_merge = 0.7, T_low=0.1, T_high=0.8, T_rel = True):
    zw2d = np.zeros(aff_uint8.shape[1:], np.uint32)
    max_id = np.uint32(0)
    for zi in range(aff_uint8.shape[1]):
        print('%d/%d' %(zi,aff_uint8.shape[1]))
        aff_z = aff_uint8[:,zi:zi+1].astype(np.float32)/255.
        seg = zwatershed.zwatershed(aff_z, T_threshes=[T_thres],
                         T_dust=T_dust, T_aff=[T_low,T_high,T_dust_merge],
                         T_aff_relative=T_rel, T_merge=T_mst_merge)[0][0][0]

        seg[seg > 0] += max_id
        max_id = seg.max()
        zw2d[zi] = seg
    return zw2d

def seg2dToseg3d(aff_uint8, seg2d, wz_thres, aff_thres = [0.1, 0.9], wz_mf = 'aff85_his256_ran255'):
        aff_thres = (np.array(aff_thres)*255).astype(np.uint8)
        wz_thres = (np.array(wz_thres)*255).astype(np.uint8)

        out = waterz.waterz(aff_uint8, wz_thres, merge_function=wz_mf, output_prefix=wz_mf,
                            aff_threshold=aff_thres, return_seg=True, fragments=seg2d)
        return out

def mergeWithSize(aff_uint8, seg3d, wz_r2_aff = 0.4, wz_r2_size = 3200, wz_r2_mst = -1, wz_dust = 200, rg_opt = 1, wz_mf = 'aff85_his256_ran255'):
    ui, uc = np.unique(seg3d, return_counts=True)
    uc_rl = np.zeros(int(ui.max()) + 1, np.uint64)
    uc_rl[ui] = uc
    rg_id, rg_sc = waterz.getRegionGraph(aff_uint8, seg3d, rg_opt, wz_mf, rebuild=True)
    rg_id = rg_id.astype(np.uint64)
    rg_sc = rg_sc.astype(np.float32) / 255.
    zwatershed.zw_merge_segments_with_function(seg3d,\
                                            1-rg_sc, rg_id[:,0], rg_id[:,1], uc_rl, wz_r2_size, wz_r2_aff, wz_dust, wz_r2_mst)
    return seg3d

def segEval(seg, gt):
    print('#seg=%d, arand=%.6f' % (len(np.unique(seg)), adapted_rand(seg, test_gt)))

if __name__ == "__main__":
    # load gt
    test_gt = readh5('test-labels.h5')
    # load input affinity 
    aff_uint8 = readh5('aff_xy.h5')
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
