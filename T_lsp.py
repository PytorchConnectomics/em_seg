import sys,yaml
from scipy.ndimage import zoom
from scipy.ndimage.morphology import binary_erosion,binary_dilation
from dataloader import *
from seg_pipeline import *
from imu.io import mkdir, readVol, writeH5

if __name__== "__main__":
    # sa zw 
    # python T_jwr15.py 0 0 1
    opt = sys.argv[1]
    job_id, job_num = 0, 1
    if len(sys.argv) > 3:
        job_id = int(sys.argv[2])
        job_num = int(sys.argv[3])
    
    pipeline = SegPipeline('/n/pfister_lab2/Lab/donglai/lib/seg/em100/data/lsp.yaml')
    pipeline.setWorkerId(job_id, job_num)
    pipeline.out_seg2d = pipeline.param_m['NUCLEUS'] + pipeline.param_m['NUCLEUS_FORMAT'] +'.' + pipeline.param_m['NUCLEUS_EXT']
    pipeline.out_seg2d_iou = pipeline.param_m['NUCLEUS'] + pipeline.param_m['NUCLEUS_IOU'] + pipeline.param_m['NUCLEUS_FORMAT'] + '_iou%s.h5'
    pipeline.out_seg3d = pipeline.param_m['NUCLEUS'] + pipeline.param_m['NUCLEUS_3D'] + pipeline.param_m['NUCLEUS_FORMAT'] + '.h5'
    mkdir(pipeline.out_seg2d_iou, 'parent')

    if opt == '0': # iou
        pipeline.seg2dToIou(False)
    elif opt == '1': # unique id
        pipeline.seg2dToGlobalId()
    elif opt == '2': # iou+unique id -> matches
        pipeline.iouToMatches()
    elif opt == '3': # iou+unique id -> matches
        pipeline.matchesTo3D()
    elif opt == '4': # low-res
        rr = np.array([2,2,2])
        seg = np.zeros(pipeline.sz//rr, np.uint16)
        for z in range(seg.shape[0]):
            seg[z] = readVol(pipeline.out_seg3d%(pipeline.ran[0][0]+z*rr[0]))[::rr[1],::rr[2]]
        writeH5('db.h5',seg)
