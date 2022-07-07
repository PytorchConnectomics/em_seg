import sys,yaml
from scipy.ndimage import zoom
from scipy.ndimage.morphology import binary_erosion,binary_dilation
from dataloader import *
from seg_pipeline import *

class MaskJWR15(MaskLoader):
    def getZslice(self, z):
        self.mask[:] = 0
        if self.name == 'bv':
            mask = self._getZslice(z) == 255
        elif self.name == 'soma':
            # shrink 1 pix
            mask = self._getZslice(z)
        if self.erosion > 0:
            mask = mask * binary_erosion(mask>0, iterations = self.erosion)
        elif self.erosion < 0:
            mask = mask * binary_dilation(mask>0, iterations = -self.erosion)

        mask = zoom(mask, self.ratio[1:], order=0)
        mask_valid = mask[max(0,self.st[1]):self.st[1]+self.sz[0], \
                   max(0,self.st[2]):self.st[2]+self.sz[1]]
        mask_st = [max(0, -self.st[1+x]) for x in range(2)] 
        self.mask[mask_st[0] : mask_st[0] + mask_valid.shape[0], \
                  mask_st[1] : mask_st[1] + mask_valid.shape[1]] = mask_valid
        return self.mask

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
