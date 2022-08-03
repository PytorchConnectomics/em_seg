import sys,yaml
from scipy.ndimage import zoom
from scipy.ndimage.morphology import binary_erosion,binary_dilation
from dataloader import *
from itertools import combinations

class SynapseP14(FileLoader):
    
    def getFilenames(self):
        zz = [0,307,615,923,1231,1539,1847,2155,2463,2771,3079]
        yy = range(0,28673,3584)
        xx = range(0,20481,2560)
        fns = []
        fns = ['/n/boslfs02/LABS/lichtman_lab/zudi/p14_syn/syn/result_%d-%d-%d-%d-%d-%d_syn.h5'%(z,zz[zi+1],y,yy[yi+1],x,xx[xi+1])]
        
        alass SynapseP14(FileLoa
class AffJWR15(AffinityLoader):
    def getZchunk(self, z):
        # yx aff
        aff_z = [[[None]*2 for x in range(len(self.ran[2]))] for y in range(len(self.ran[1]))]
        for xi, x in enumerate(self.ran[2]):
            for yi, y in enumerate(self.ran[1]):
                for ki,k in enumerate('yx'):
                    tmp = h5py.File(self.filename % (x,y,z,k),'r')
                    aff_z[yi][xi][ki] = tmp[list(tmp)[0]]
        return aff_z

    def getZslice(self, zchunk, zis):
        aff = np.zeros([3, len(zis), self.sz[1], self.sz[2]], np.uint8)
        for xi,x in enumerate(self.ran[2]):
            for yi,y in enumerate(self.ran[1]):
                for ki in range(2):
                    aff[1+ki,:,yi*self.chunk_sz[1]:(yi+1)*self.chunk_sz[1],xi*self.chunk_sz[2]:(xi+1)*self.chunk_sz[2]] = \
                        np.array(zchunk[yi][xi][ki][zis[0]:zis[-1]+1])
        return aff



if __name__== "__main__":
    # python T_jwr15.py 0 0 1
    opt = sys.argv[1]
    job_id = int(sys.argv[2])
    job_num = int(sys.argv[3])
    
    pipeline = SegPipeline('/n/pfister_lab2/Lab/donglai/lib/seg/em100/data/jwr15.yaml', AffJWR15, MaskJWR15)
    pipeline.setWorkerId(job_id, job_num)
    if opt == '0': # build waterz agglomeration
        rg_id, rg_score = waterz.getRegionGraph(np.zeros([3,1,10,10],np.uint8), np.zeros([1,10,10],np.uint8), 1, pipeline.param_s['SEG2D']['RG_MERGE_FUNC'], rebuild=True)
    elif opt == '1': # affinity to 2D zwatershed for whole slice
        pipeline.affinityToSeg2D()
    elif opt == '1.1': # seg2D to IoU
        pipeline.seg2DToIou()
