import sys,yaml
from scipy.ndimage import zoom
from scipy.ndimage.morphology import binary_erosion,binary_dilation
from dataloader import *
from seg_pipeline import *

class MaskR0(MaskLoader):
    pass
if __name__== "__main__":
    # sa zw 
    # python T_jwr15.py 0 0 1
    opt = sys.argv[1]
    job_id, job_num = 0, 1
    if len(sys.argv) > 3:
        job_id = int(sys.argv[2])
        job_num = int(sys.argv[3])
    
    pipeline = SegPipeline('/n/pfister_lab2/Lab/donglai/lib/seg/em100/data/r0.yaml', None, MaskR0)
    pipeline.setWorkerId(job_id, job_num)
    pipeline.setOutputFolder(pipeline.param_m['BLOOD_VESSEL_OUT'])
    chunk_sz=[1101,0,0]
    chunk_sz=[3302,0,0]
    if opt == '0': # build waterz agglomeration
        pipeline.maskToCC('BLOOD_VESSEL')
    elif opt == '1.1': # maskcc to IoU
        pipeline.seg2DToIou()
    elif opt == '1.2': # seg2D to rg-z
        pipeline.seg2DToRgZByIoU()
    elif opt == '1.3': # group seg2D by chunk
        pipeline.seg2DToChunk(do_soma=False, thres = -0.3, chunk_sz=chunk_sz)
    elif opt == '1.32': # group seg2D chunks: debug
        pipeline.seg2DToChunkDecode(do_soma=False, chunk_sz=chunk_sz, zstep=1)
    elif opt == '1.33': # group seg2D chunks: debug
        out = np.zeros([425,400,400],np.uint32)
        import pdb; pdb.set_trace()
        for z in range(out.shape[0]):
            sn = pipeline.out_seg2d_db%(1+z*8)+'.h5'
            if os.path.exists(sn):
                out[z] = readH5(sn)[::4,::4]
        writeH5('db.h5', out)
    """
    im_id = pipeline.im_id
    import shutil
    D0='/n/boslfs02/LABS/lichtman_lab/donglai/R0/coarse_seg/coarse_seg_cc/'
    for z in range(len(im_id)-2,-1,-1): 
        #shutil.move(D0+'%04d.h5'%im_id[z], D0+'%04d.h5'%(im_id[z+1]))
        for k in 'fb':
            if os.path.exists(D0+'%04d_iou%s.h5'%(im_id[z],k)):
                shutil.move(D0+'%04d_iou%s.h5'%(im_id[z],k), D0+'%04d_iou%s.h5'%(im_id[z+1],k))
    """
