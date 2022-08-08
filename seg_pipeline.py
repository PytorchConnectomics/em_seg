import os,yaml
import waterz
import cc3d
import numpy as np
from imu.io import readH5, writeH5, mkdir

class SegPipeline:
    def __init__(self, param_file, aff_class, mask_class):
        self.loadParam(param_file)

        aff_oset = self.param_a['OFFSET']
        aff_st = [self.ran[x][0]-aff_oset[x] for x in range(3)]
        self.aff_d = aff_class(self.sz, aff_st, self.chunk_sz, self.param_a['FILENAME'])
        self.bv_d, self.soma_d, self.border_d = None, None, None
        if 'BLOOD_VESSEL' in self.param_m:
            oset = self.param_m['BLOOD_VESSEL_OFFSET']
            st = [self.ran[x][0]-oset[x] for x in range(3)]
            self.bv_d = mask_class('bv', st, self.sz[1:], self.param_m['BLOOD_VESSEL'], self.param_m['BLOOD_VESSEL_EROSION'], self.param_m['BLOOD_VESSEL_RATIO'])
        if 'SOMA' in self.param_m:
            oset = self.param_m['SOMA_OFFSET']
            st = [self.ran[x][0]-oset[x] for x in range(3)]
            self.soma_d = mask_class('soma', st, self.sz[1:], self.param_m['SOMA'], self.param_m['SOMA_EROSION'], self.param_m['SOMA_RATIO'])
        if 'BORDER' in self.param_m:
            oset = self.param_m['BORDER_OFFSET']
            st = [self.ran[x][0]-oset[x] for x in range(3)]
            self.border_d = mask_class('border', oset, self.sz[1:], self.param_m['BORDER'], self.param_m['BORDER_EROSION'], self.param_m['BORDER_RATIO'])

    def loadParam(self, param_file):
        param = yaml.load(open(param_file))
        self.param_d = param['DATA']
        self.param_m = param['MASK']
        self.param_a = param['AFF']
        self.param_s = param['SEG']
        self.param_ws = param
        self.ran = [self.param_d['ZRAN'], self.param_d['YRAN'], self.param_d['XRAN']]
        # global canvas
        self.sz_all = self.param_d['SIZE']
        # output size
        self.sz = [self.ran[x][1]-self.ran[x][0] for x in range(3)]
        # affinity
        self.chunk_sz = self.param_a['CHUNK_SIZE']

        self.dtype = eval('np.uint%d' % self.param_s['MAX_BIT'])
        self.out_seg2d = self.param_s['OUTPUT_FOLDER'] + self.param_s['SEG2D']['OUTPUT']
        self.out_seg2d_iou = self.param_s['OUTPUT_FOLDER'] + self.param_s['SEG2D']['OUTPUT_IOU']
        self.out_seg2d_rgz = self.param_s['OUTPUT_FOLDER'] + self.param_s['SEG2D']['OUTPUT_RGZ']
        self.out_seg2d_chunk = self.param_s['OUTPUT_FOLDER'] + self.param_s['SEG2D']['OUTPUT_CHUNK']
        self.out_seg2d_chunk_db = self.param_s['OUTPUT_FOLDER'] + self.param_s['SEG2D']['OUTPUT_CHUNK_DB']
        self.out_seg2d_all = self.param_s['OUTPUT_FOLDER'] + self.param_s['SEG2D']['OUTPUT_ALL']
        mkdir(self.out_seg2d, 'parent')

    def setWorkerId(self, job_id, job_num):
        self.job_id = job_id
        self.job_num = job_num
    
    def getOutputSeg2D(self, z):
        return self.param_s['SEG_2D']['OUTPUT'] % z

    def affinityToSeg2D(self):
        for zchunk in range(self.ran[0][0],self.ran[0][1],self.chunk_sz[0])[self.job_id::self.job_num]:
            aff_zchunk = self.aff_d.getZchunk(zchunk)
            for z in range(self.chunk_sz[0]):
                zz = zchunk + z
                sn = self.out_seg2d % (zz)
                if not os.path.exists(sn):
                    aff = self.aff_d.getZslice(aff_zchunk, [z])
                    # writeH5('db.h5', aff)
                    #aff = readH5('db.h5')
                    mask_bv, mask_soma = None, None
                    mask_bv = None if self.bv_d is None else self.bv_d.getZslice(zz)
                    mask_soma = None if self.soma_d is None else self.soma_d.getZslice(zz)
                    # writeH5('db.h5', mask_bv.astype(np.uint8))
                    mask_border = None if self.border_d is None else self.border_d.getZslice(zz)
                    seg, soma_rl = self._affinityToSeg2D(aff, mask_bv, mask_soma, mask_border)
                    writeH5(sn, [seg, soma_rl, np.array([seg.max()])], ['main','soma_rl', 'max'])

    def seg2DToIou(self):
        for zz in range(self.ran[0][0],self.ran[0][1]-1)[self.job_id::self.job_num]:
            sn1 = self.out_seg2d % (zz)
            sn2 = self.out_seg2d % (zz+1)
            sn_f = self.out_seg2d_iou % (zz, 'f')
            sn_b = self.out_seg2d_iou % (zz+1, 'b')
            if not os.path.exists(sn_b):
                seg1 = readH5(sn1, 'main')[0] 
                seg2 = readH5(sn2, 'main')[0] 
                bb1 = self._get_bb_all2d(seg1)
                bb2 = self._get_bb_all2d(seg2)
                iou = self._seg_iou2d(seg1, seg2, bb1=bb1, bb2=bb2)
                writeH5(sn_f, iou)
                iou = self._seg_iou2d(seg2, seg1, bb1=bb2, bb2=bb1)
                writeH5(sn_b, iou)

    def seg2DToRgZ(self):
        rg_m1_func, rg_m1_aff = self.param_s['SEG2D']['RG_MERGE_FUNC'], self.param_s['SEG2D']['RG_MERGE_AFF']
        for zchunk in range(self.ran[0][0],self.ran[0][1],self.chunk_sz[0])[self.job_id::self.job_num]:
            aff_zchunk = self.aff_d.getZchunk(zchunk, True)
            for z in range(self.chunk_sz[0]):
                zz = z + zchunk
                if zchunk == self.ran[0][1] - self.chunk_sz[0] and z == self.chunk_sz[0]-1:
                    continue
                sn1 = self.out_seg2d % (zz)
                sn2 = self.out_seg2d % (zz+1)
                sno = self.out_seg2d_rgz % (zz)
                if not os.path.exists(sno):
                    seg = np.concatenate([readH5(sn1, 'main'), readH5(sn2, 'main')], axis = 0) 
                    seg_m = seg[0].max()
                    # avoid rg_id switch position due to minmax
                    seg[1][seg[1] > 0] += seg_m 
                    if (z+1) % self.chunk_sz[0] == 0: # last slice
                        aff_zchunk = self.aff_d.getZchunk(zchunk + self.chunk_sz[0], True)
                        aff = self.aff_d.getZslice(aff_zchunk, [0], True)
                    else:
                        aff = self.aff_d.getZslice(aff_zchunk, [z+1], True)
                    aff = np.concatenate([np.zeros_like(aff), aff], axis=1)
                    # only z-aff
                    rg_id, rg_score = waterz.getRegionGraph(aff, seg, 3, rg_m1_func, rebuild=False)
                    rg_id[:,1] -= seg_m 
                    # rg_id.max(axis=0),[seg[0].max(),seg[1].max()]
                    writeH5(sno, [rg_id, rg_score], ['id', 'score'])

    def seg2DToChunk(self):
        rg_m1_aff = self.param_s['SEG2D']['RG_MERGE_AFF']
        for zchunk in range(self.ran[0][0],self.ran[0][1],self.chunk_sz[0])[self.job_id::self.job_num]:
            snc = self.out_seg2d_chunk % zchunk 
            if not os.path.exists(snc):
                cc = np.zeros(1+self.chunk_sz[0], np.uint32)
                for z in range(self.chunk_sz[0]):
                    zz = z + zchunk
                    sn = self.out_seg2d % (zz)
                    cc[1+z] = readH5(sn, 'max')
                cc = np.cumsum(cc).astype(np.uint32)
                
                matches = np.zeros([0,2], np.uint32)
                soma_m = 0
                #for z in range(14):
                for z in range(self.chunk_sz[0]):
                    zz = z + zchunk
                    sno = self.out_seg2d_rgz % (zz)
                    rg_id, rg_score = readH5(sno, ['id', 'score'])

                    sn = self.out_seg2d % (zz)
                    soma_rl = readH5(sn, 'soma_rl')
                    rl = np.arange(soma_rl.max()+1).astype(np.uint32)
                    rl[soma_rl[:,1]] = cc[-1] + soma_rl[:,0]
                    rg_id[:,0] = rl[rg_id[:,0]] + cc[z]

                    sn = self.out_seg2d % (zz+1)
                    soma_rl = readH5(sn, 'soma_rl')
                    rl = np.arange(soma_rl.max()+1).astype(np.uint32)
                    rl[soma_rl[:,1]] = cc[-1] + soma_rl[:,0]
                    rg_id[:,1] = rl[rg_id[:,1]] + cc[z+1]

                    matches = np.vstack([matches, rg_id[rg_score <= rg_m1_aff]])
                    soma_m = max(soma_m, soma_rl[:,0].max())
                # remove pairs that lead to soma seg to merge
                jj = waterz.somaBFS(matches, np.unique(matches[matches>cc[-1]]))
                jj = np.vstack([jj, [cc[-1]+soma_m, cc[-1]+soma_m]]).astype(np.uint32)
                # assign soma_id to big ids 
                relabel = waterz.merge_id(jj[:,0], jj[:,1], id_thres = cc[-1]+1)
                uid = np.unique(relabel[:cc[-1]+1])
                rl = np.zeros(uid.max()+1, np.uint32)
                rl[uid] = np.arange(1, 1+len(uid))
                relabel = rl[relabel[:cc[-1]+1]]
                writeH5(snc, [cc, relabel, np.array([1+len(uid), soma_m])], ['count', 'relabel', 'sid'])

    def seg2DToChunkDecode(self):
        for zchunk in range(self.ran[0][0],self.ran[0][1],self.chunk_sz[0])[self.job_id::self.job_num]:
            snc = self.out_seg2d_chunk % zchunk 
            count, relabel, sid = readH5(snc)
            #import pdb; pdb.set_trace()
            for z in range(self.chunk_sz[0]):
                zz = z + zchunk
                print(zz)
                snd = self.out_seg2d_chunk_db % zz 
                sno = self.out_seg2d_rgz % (zz)
                rg_id, rg_score = readH5(sno, ['id', 'score'])

                sn = self.out_seg2d % (zz)
                soma_rl = readH5(sn, 'soma_rl')
                rl = np.arange(soma_rl.max()+1).astype(np.uint32)
                rl[soma_rl[:,1]] = count[-1] + soma_rl[:,0]

                seg = rl[readH5(sn, 'main')]
                seg[(seg>0)*(seg<=count[-1])] = relabel[seg[(seg>0)*(seg<=count[-1])]+count[z]]
                writeH5(snd, seg)
 
    def segChunkToAll(self):
        rg_m1_aff = param['RG_MERGE_AFF']
        sna = self.out_seg2d_all
        if not os.path.exists(sna):
            cc = np.zeros(1+(self.ran[0][1]-self.ran[0][0])//self.chunk_sz[0])
            for zi,zchunk in enumerate(range(self.ran[0][0],self.ran[0][1],self.chunk_sz[0])):
                snc = self.out_seg2d_chunk % zchunk 
                cc[zi+1] = readH5(snc, 'sid')[0]
            cc = np.cumsum(cc)

            matches = np.zeros([0,2])
            soma_m = 0
            for zi,zchunk in enumerate(range(self.ran[0][0], self.ran[0][1]-self.chunk_sz[0], self.chunk_sz[0])):
                sno = self.out_seg2d_rgz % (zchunk)
                rg_id, rg_score = readH5(sno, ['id', 'score'])

                sn = self.out_seg2d % (zchunk+self.chunk_sz[0]-1)
                soma_rl = readH5(sn, 'soma_rl')
                rl = np.arange(soma_rl.max()+1).astype(np.uint32)
                rl[soma_rl[:,1]] = cc[-1] + soma_rl[:,0]
                snc = self.out_seg2d_chunk % zchunk 
                count, relabel = readH5(snc, ['count','relabel'])
                gid = rg_id[:,0] < soma_rl[:,1].min()
                rg_id[gid,0] = relabel[rg_id[gid,0]+count[-1]] + cc[zi]
                rg_id[gid==0, 0] = rl[rg_id[gid==0,0]]
                soma_m = max(soma_m, readH5(snc, 'sid')[1])

                sn = self.out_seg2d % (zchunk+self.chunk_sz[0])
                soma_rl = readH5(sn, 'soma_rl')
                rl = np.arange(soma_rl.max()+1).astype(np.uint32)
                rl[soma_rl[:,1]] = cc[-1] + soma_rl[:,0]
                snc = self.out_seg2d_chunk % (zchunk+self.chunk_sz[0]) 
                count, relabel = readH5(snc, ['count','relabel'])
                gid = rg_id[:,1] < soma_rl[:,1].min()
                rg_id[gid,1] = relabel[rg_id[gid,1]+count[0]] + cc[zi+1]
                rg_id[gid==0, 1] = rl[rg_id[gid==0, 1]]
                soma_m = max(soma_m, readH5(snc, 'sid')[1])

                matches = np.vstack([matches, rg_id[rg_score <= rg_m1_aff]])
            # remove pairs that lead to soma seg to merge
            jj = waterz.somaBFS(matches, np.unique(matches[matches>cc[-1]]))
            # assign soma_id to big ids 
            jj = np.vstack([jj, [cc[-1]+soma_m, cc[-1]+soma_m]])
            # assign soma_id to big ids 
            relabel = waterz.merge_id(jj[:,0], jj[:,1], id_thres = cc[-1]+1)
            uid = np.unique(relabel[:cc[-1]+1])
            rl = np.zeros(uid.max()+1, np.uint32)
            rl[uid] = np.arange(1, 1+len(uid))
            relabel = rl[relabel[:cc[-1]+1]]
            writeH5(sna, [cc, relabel, np.array([1+len(uid), soma_m])], ['count', 'relabel', 'sid'])

    def _affinityToSeg2D(self, aff, mask_bv=None, mask_soma=None, mask_border=None):
        # aff: 3x1xHxW
        param = self.param_s['SEG2D']
        ws_low, ws_nb, ws_dust = param['AFF_LOW_THRES'], param['WS_NB_SIZE'], param['DUST_SIZE']
        rg_m1_func, rg_m1_aff = param['RG_MERGE_FUNC'], param['RG_MERGE_AFF']
        rg_m2_size, rg_m2_aff = param['RG_MERGE2_SIZE'], param['RG_MERGE2_AFF']

        # 1. set low-aff region to background
        aff[aff < ws_low] = 0
        # get external mask
        if mask_bv is not None:
            aff[:,0] = aff[:,0] * (1 - mask_bv)
        if mask_border is not None:
            aff[:,0] = aff[:,0] * (1 - mask_border)

        # initial watershed
        seg = waterz.watershed(aff, label_nb = np.ones([ws_nb,ws_nb]), bg_thres = 1 - ws_low/255.)
        seg_m = seg.max() + 1

        # snap to soma id
        soma_rl = np.zeros([1,2])
        if mask_soma is not None and mask_soma.any():
            soma_ids = np.unique(mask_soma[mask_soma>0])
            rl = np.arange(seg_m).astype(self.dtype)
            # need to split sometimes
            soma_rl = np.zeros([len(soma_ids),2],int)
            for i,soma_id in enumerate(soma_ids):
                ii = np.unique(seg[0][mask_soma==soma_id])
                # make sure no overlap among somas 
                ii = ii[(ii>0)*(ii<seg_m)]
                # remove small olap
                if len(ii) > 0:
                    rl[ii] = seg_m + i
                    soma_rl[i] = [soma_id, seg_m + i]
            # merge soma regions
            seg = rl[seg]
            
        # compute region graph
        rg_id, rg_score = waterz.getRegionGraph(aff, seg, 1, rg_m1_func, rebuild=False)

        # merge 1: conservative
        jj = rg_id[rg_score <= rg_m1_aff]
        if soma_rl[:,1].any():
            # remove pairs that lead to soma seg to merge
            jj = waterz.somaBFS(jj, soma_rl[soma_rl[:,1]>0, 1])

        out = waterz.merge_id(jj[:,0], jj[:,1], id_thres = seg_m)
        out_l = len(out)
        seg[seg < out_l] = out[seg[seg < out_l]]
        
        # merge 2: small size with higher thres [sorted aff]
        # assume it won't cause merge among neurons
        ui, uc = np.unique(seg[0], return_counts=True)
        uc_rl = np.zeros(int(ui.max()) + 1, self.dtype)
        uc_rl[ui] = uc
        gid =  (rg_score <= rg_m2_aff)*((uc_rl[rg_id] <= rg_m2_size).sum(axis=1)>0)
        jj = rg_id[gid]
        jj_sc = rg_score[gid]
        sid = np.argsort(jj_sc)
        out = waterz.merge_id(jj[sid,0], jj[sid,1], jj_sc[sid], uc_rl, seg_m, rg_m2_aff, rg_m2_size, ws_dust)
        out_l = len(out)
        seg[seg < out_l] = out[seg[seg < out_l]]

        return seg, soma_rl

    def _seg_iou2d(self, seg1, seg2, ui0=None, bb1=None, bb2=None):
        # bb1/bb2: first column of indexing, last column of size
        
        if bb1 is None:
            ui,uc = np.unique(seg1,return_counts=True)
            uc=uc[ui>0];ui=ui[ui>0]
        else:
            ui = bb1[:,0]
            uc = bb1[:,-1]

        if bb2 is None:
            ui2, uc2 = np.unique(seg2,return_counts=True)
        else:
            ui2 = bb2[:,0]
            uc2 = bb2[:,-1]

        if bb1 is None:
            if ui0 is None:
                bb1 = self._get_bb_all2d(seg1, uid=ui)
                ui0 = ui
            else:
                bb1 = self._get_bb_all2d(seg1, uid=ui0)
        else:
            if ui0 is None:
                ui0 = ui
            else:
                # make sure the order matches..
                bb1 = bb1[np.in1d(bb1[:,0], ui0)]
                ui0 = bb1[:,0] 

        out = np.zeros((len(ui0),5),int)
        out[:,0] = ui0
        out[:,2] = uc[np.in1d(ui,ui0)]

        for j,i in enumerate(ui0):
            bb= bb1[j, 1:]
            ui3,uc3 = np.unique(seg2[bb[0]:bb[1]+1,bb[2]:bb[3]+1]*(seg1[bb[0]:bb[1]+1,bb[2]:bb[3]+1]==i),return_counts=True)
            uc3[ui3==0] = 0
            if (ui3>0).any():
                out[j,1] = ui3[np.argmax(uc3)]
                out[j,3] = uc2[ui2==out[j,1]]
                out[j,4] = uc3.max()
        return out

    def _get_bb_all2d(self, seg, do_count=False, uid=None):
        sz = seg.shape
        assert len(sz)==2
        if uid is None:
            uid = np.unique(seg)
            uid = uid[uid>0]
        if len(uid) == 0:
            return np.zeros((1,5+do_count),dtype=np.uint32)

        um = uid.max()
        out = np.zeros((1+int(um),5+do_count),dtype=np.uint32)
        out[:,0] = np.arange(out.shape[0])
        out[:,1] = sz[0]
        out[:,3] = sz[1]
        # for each row
        rids = np.where((seg>0).sum(axis=1)>0)[0]
        for rid in rids:
            sid = np.unique(seg[rid])
            sid = sid[(sid>0)*(sid<=um)]
            out[sid,1] = np.minimum(out[sid,1],rid)
            out[sid,2] = np.maximum(out[sid,2],rid)
        cids = np.where((seg>0).sum(axis=0)>0)[0]
        for cid in cids:
            sid = np.unique(seg[:,cid])
            sid = sid[(sid>0)*(sid<=um)]
            out[sid,3] = np.minimum(out[sid,3],cid)
            out[sid,4] = np.maximum(out[sid,4],cid)

        if do_count:
            ui,uc = np.unique(seg,return_counts=True)
            out[ui,-1]=uc
        return out[uid]
