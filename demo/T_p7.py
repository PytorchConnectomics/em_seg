import sys,yaml
from scipy.ndimage import zoom
from scipy.ndimage.morphology import binary_erosion,binary_dilation
from dataloader import *
from itertools import combinations

class MaskR0(MaskLoader):
    pass

if __name__== "__main__":
    # python T_jwr15.py 0 0 1
    opt = sys.argv[1]
    job_id = int(sys.argv[2])
    job_num = int(sys.argv[3])
    
    pipeline = SegPipeline('/n/pfister_lab2/Lab/donglai/lib/seg/em100/data/p7.yaml', None, MaskR0)
    pipeline.setWorkerId(job_id, job_num)
    pipeline.setOutputFolder(pipeline.param_m['BLOOD_VESSEL_OUT'])
    chunk_sz=[1101,0,0]
    chunk_sz=[3302,0,0]
    if opt == '0': # build waterz agglomeration
        pipeline.maskToCC('BLOOD_VESSEL')
    elif opt == '1.1': # maskcc to IoU
        pipeline.seg2DToIou()
    
    pipeline = SegPipeline('/n/pfister_lab2/Lab/donglai/lib/seg/em100/data/jwr15.yaml', AffJWR15, MaskJWR15)
    pipeline.setWorkerId(job_id, job_num)
    if opt == '0': # build waterz agglomeration
        rg_id, rg_score = waterz.getRegionGraph(np.zeros([3,1,10,10],np.uint8), np.zeros([1,10,10],np.uint8), 1, pipeline.param_s['SEG2D']['RG_MERGE_FUNC'], rebuild=True)
    elif opt == '1': # affinity to 2D zwatershed for whole slice
        pipeline.affinityToSeg2D()
    elif opt == '1.1': # seg2D to IoU
        pipeline.seg2DToIou()
