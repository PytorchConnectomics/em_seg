import os,yaml
import cc3d
import numpy as np
import h5py
from imu.io import readVol, readH5,writeH5, mkdir, get_bb_all2d
from imu.seg import seg_iou2d, iouToMatches, seg2dToGlobalId, merge_id, seg2dToGlobal
from tqdm import tqdm

class SegPipeline:
    def __init__(self, param_file, aff_class=None, mask_class=None):
        self.loadParam(param_file)

        if aff_class is not None:
            aff_oset = self.param_a['OFFSET']
            aff_st = [self.ran[x][0]-aff_oset[x] for x in range(3)]
            self.aff_d = aff_class(self.sz, aff_st, self.chunk_sz, self.param_a['FILENAME'])
        # low-res seg mask
        self.bv_d, self.soma_d, self.border_d = None, None, None
        oset,ratio,erosion = [0,0,0],[1,1,1],0
        if 'BLOOD_VESSEL' in self.param_m:
            oset2 = self.param_m['BLOOD_VESSEL_OFFSET'] if 'BLOOD_VESSEL_OFFSET' in self.param_m else oset
            erosion2 = self.param_m['BLOOD_VESSEL_EROSION'] if 'BLOOD_VESSEL_EROSION' in self.param_m else erosion
            ratio2 = self.param_m['BLOOD_VESSEL_RATIO'] if 'BLOOD_VESSEL_RATIO' in self.param_m else ratio
            st = [self.ran[x][0]-oset2[x] for x in range(3)]
            self.bv_d = mask_class('bv', st, self.sz[1:], self.param_m['BLOOD_VESSEL'], erosion2, ratio2)
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

        self.ran = [self.param_d['ZRAN'], self.param_d['YRAN'], self.param_d['XRAN']]
        if 'IMID' in self.param_d:
            self.im_id = np.loadtxt(self.param_d['IMID']).astype(int)
        else:
            self.im_id = np.arange(self.ran[0][0],self.ran[0][1]).astype(int)
        # global canvas
        self.sz_all = self.param_d['SIZE']
        # output size
        self.sz = [self.ran[x][1]-self.ran[x][0] for x in range(3)]
        self.output_root =  self.param_d['OUTPUT_ROOT'] if 'OUTPUT_ROOT' in self.param_d else './'
        num_digit = int(np.ceil(np.log10(self.ran[0][1])))
        self.output_format = '%0'+str(num_digit)+'d'
        self.setOutputFolder()
        mkdir(self.output_folder)

        if 'AFF' in param:
            # affinity
            self.param_a = param['AFF']
            self.chunk_sz = self.param_a['CHUNK_SIZE']
        if 'MASK' in param:
            self.param_m = param['MASK']
        if 'SEG' in param:
            self.param_s = param['SEG']
            self.dtype = eval('np.uint%d' % self.param_s['MAX_BIT'])
        self.param_ws = param

    def setOutputFolder(self, folder= 'seg/'):
        self.output_folder = self.output_root + folder
        self.out_seg2d = self.output_folder + self.output_format + '.h5'
        self.out_seg2d_iou = self.output_folder + self.output_format + '_iou%s.h5'
        self.out_seg2d_rgz = self.output_folder + self.output_format + '_rgz.h5'
        self.out_seg2d_chunk = self.output_folder + self.output_format + '_chunk.h5'
        self.out_seg2d_db = self.output_folder[:-1]+'_db/' + self.output_format 
        #mkdir(self.out_seg2d_db, 'parent')

    def setWorkerId(self, job_id, job_num):
        self.job_id = job_id
        self.job_num = job_num
    
    def maskToCC(self, mask_type='BLOOD_VESSEL'):
        for i,z in enumerate(self.im_id):
            if i % self.job_num != self.job_id:
                continue
            sn = self.param_d['OUTPUT_FOLDER']
            if mask_type == 'BLOOD_VESSEL':
                sn += self.param_m['BLOOD_VESSEL_OUT'] % (z)
                if not os.path.exists(sn):
                    print(i,z)
                    mask = cc3d.connected_components(self.bv_d._getZslice(i), connectivity=4) 
                    bb = get_bb_all2d(mask, True)
                    writeH5(sn,[mask, bb],['main','bb'])

    def affinityToSeg2D(self):
        for zchunk in range(self.ran[0][0],self.ran[0][1],self.chunk_sz[0])[self.job_id::self.job_num]:
            aff_zchunk = self.aff_d.getZchunk(zchunk)
            for z in range(self.chunk_sz[0]):
                zz = zchunk + z
                sn = self.output_seg % (zz)
                if not os.path.exists(sn):
                    aff = self.aff_d.getZslice(aff_zchunk, [z])
                    # writeH5('db.h5', aff)
                    #aff = readVol('db.h5')
                    mask_bv, mask_soma = None, None
                    mask_bv = None if self.bv_d is None else self.bv_d.getZslice(zz)
                    mask_soma = None if self.soma_d is None else self.soma_d.getZslice(zz)
                    # writeH5('db.h5', mask_bv.astype(np.uint8))
                    mask_border = None if self.border_d is None else self.border_d.getZslice(zz)
                    seg, soma_rl = self._affinityToSeg2D(aff, mask_bv, mask_soma, mask_border)
                    writeH5(sn, [seg, soma_rl, np.array([seg.max()])], ['main','soma_rl', 'max'])

    def seg2DToIou(self, do_back=True):
        for i,zz in enumerate(self.im_id[:-1]):
            if i% self.job_num != self.job_id:
                continue
            sn2 = self.out_seg2d % (self.im_id[i+1])
            sn_f = self.out_seg2d_iou % (zz, 'f')
            sn_b = sn_f
            if do_back:
                sn_b = self.out_seg2d_iou % (self.im_id[i+1], 'b')
            if not os.path.exists(sn_b):
                print(sn1,sn2)
                seg1 = readVol(sn1).squeeze()
                seg2 = readVol(sn2).squeeze()
                if sn1[-3:]=='h5' and 'bb' in list(h5py.File(sn1,'r')):
                    bb1 = readVol(sn1, 'bb')
                    bb2 = readVol(sn2, 'bb')
                else:
                    bb1 = get_bb_all2d(seg1, True)
                    bb2 = get_bb_all2d(seg2, True)
                iou = seg_iou2d(seg1, seg2, bb1=bb1, bb2=bb2)
                writeH5(sn_f, iou)
                if do_back:
                    iou = seg_iou2d(seg2, seg1, bb1=bb2, bb2=bb1)
                    writeH5(sn_b, iou)

    def seg2dToGlobalId(self):
        out = seg2dToGlobalId(self.out_seg2d, self.im_id)
        np.savetxt(os.path.dirname(self.out_seg2d)+'uid.txt', out, '%d')

    def iouToMatches(self, iou_type='f'):
        global_id = np.loadtxt(os.path.dirname(self.out_seg2d)+'uid.txt').astype(int) 
        fn_iou = self.out_seg2d_iou.replace('%s', iou_type)
        out = iouToMatches(fn_iou, self.im_id, global_id)
        writeH5(os.path.dirname(self.out_seg2d_iou)+'match.h5', out)

    def matchesTo3D(self):
        matches = readH5(os.path.dirname(self.out_seg2d_iou)+'match.h5').astype(np.uint32)
        mapping = merge_id(matches[0], matches[1])
        global_id = np.loadtxt(os.path.dirname(self.out_seg2d)+'uid.txt').astype(int) 
        mkdir(self.out_seg3d, 'parent')
        
        for i,zz in enumerate(tqdm(self.im_id)):
            if i% self.job_num != self.job_id:
                continue
            sn = self.out_seg3d % self.im_id[i]
            if not os.path.exists(sn):
                seg = readVol(self.out_seg2d % self.im_id[i])
                seg[seg>0] += global_id[i] 
                writeH5(sn, seg2dToGlobal(seg, mapping))

    def seg2DToRgZByIoU(self):
        for i,zz in enumerate(tqdm(self.im_id[:-1])):
            if i % self.job_num != self.job_id:
                continue
            sno = self.out_seg2d_rgz % (zz)
            if not os.path.exists(sno):
                sn_f = self.out_seg2d_iou % (zz, 'f')
                sn_b = self.out_seg2d_iou % (self.im_id[i+1], 'b')
                iou_f = readVol(sn_f)
                iou_b = readVol(sn_b)
                # only z-aff
                rg_id = np.vstack([iou_f[:,:2], iou_b[:,1::-1]])
                rg_score = np.hstack([iou_f[:,-1].astype(float)/iou_f[:,-3:-1].max(axis=1), iou_b[:,-1].astype(float)/iou_b[:,-3:-1].max(axis=1)])
                writeH5(sno, [rg_id, rg_score], ['id', 'score'])

    def seg2DToRgZByAff(self):
        import waterz
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
                    seg = np.concatenate([readVol(sn1, 'main'), readVol(sn2, 'main')], axis = 0) 
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

    def seg2DToChunk(self, do_soma=True, thres = None, chunk_sz = None):
        import waterz
        if thres is None:
            thres = self.param_s['SEG2D']['RG_MERGE_AFF']
        if chunk_sz is None:
            chunk_sz = self.chunk_sz

        for zchunk in range(self.ran[0][0],self.ran[0][1],chunk_sz[0])[self.job_id::self.job_num]:
            snc = self.out_seg2d_chunk % zchunk 
            if not os.path.exists(snc):
                cc = np.zeros(1+chunk_sz[0], np.uint32)
                num_slice = min(chunk_sz[0], self.ran[0][1]- zchunk)
                print(zchunk,num_slice)
                for z in range(num_slice):
                    zz = self.im_id[z + zchunk]
                    sn = self.out_seg2d % (zz)
                    if 'max' in h5py.File(sn,'r'):
                        num = readVol(sn, 'max')
                    elif 'bb' in h5py.File(sn,'r'):
                        num = readVol(sn, 'bb')[-1,0]
                    else:
                        num = readVol(sn, 'main').max()
                    cc[1+z] = num 
                cc = np.cumsum(cc).astype(np.uint32)
                
                matches = np.zeros([0,2], np.uint32)
                soma_m = 0
                #for z in range(14):
                for z in range(num_slice):
                    if zchunk == self.ran[0][1] - num_slice and z == num_slice - 1:
                        continue
                    zz = self.im_id[z + zchunk]
                    sno = self.out_seg2d_rgz % (zz)
                    rg_id, rg_score = readVol(sno, ['id', 'score'])

                    if do_soma:
                        sn = self.out_seg2d % (zz)
                        soma_rl = readVol(sn, 'soma_rl')
                        rl = np.arange(soma_rl.max()+1).astype(np.uint32)
                        rl[soma_rl[:,1]] = cc[-1] + soma_rl[:,0]
                        rg_id[:,0] = rl[rg_id[:,0]] + cc[z]
                        soma_m = max(soma_m, soma_rl[:,0].max())

                        sn = self.out_seg2d % (zz+1)
                        soma_rl = readVol(sn, 'soma_rl')
                        rl = np.arange(soma_rl.max()+1).astype(np.uint32)
                        rl[soma_rl[:,1]] = cc[-1] + soma_rl[:,0]
                        rg_id[:,1] = rl[rg_id[:,1]] + cc[z+1]
                        soma_m = max(soma_m, soma_rl[:,0].max())
                    else:
                        rg_id[:,0][rg_id[:,0]>0] += cc[z]
                        rg_id[:,1][rg_id[:,1]>0] += cc[z+1]

                    if thres < 0:
                        matches = np.vstack([matches, rg_id[rg_score >= -thres]])
                    else:
                        matches = np.vstack([matches, rg_id[rg_score <= thres]])
                if do_soma:
                    # remove pairs that lead to soma seg to merge
                    jj = waterz.somaBFS(matches, np.unique(matches[matches>cc[-1]]))
                    # assign soma_id to big ids 
                    jj = np.vstack([jj, [cc[-1]+soma_m, cc[-1]+soma_m]]).astype(np.uint32)
                else:
                    jj = np.vstack([matches, [cc[-1], cc[-1]]]).astype(np.uint32)
                    relabel = waterz.merge_id(jj[:,0], jj[:,1])

                uid = np.unique(relabel[:cc[-1]+1])
                rl = np.zeros(uid.max()+1, np.uint32)
                rl[uid] = np.arange(1, 1+len(uid))
                relabel = rl[relabel[:cc[-1]+1]]

                writeH5(snc, [cc, relabel, np.array([1+len(uid), soma_m])], ['count', 'relabel', 'sid'])

    def seg2DToChunkDecode(self, do_soma=True, chunk_sz=None,zstep=1):
        if chunk_sz is None:
            chunk_sz = self.chunk_sz
        for zchunk in range(self.ran[0][0],self.ran[0][1],chunk_sz[0])[self.job_id::self.job_num]:
            snc = self.out_seg2d_chunk % zchunk 
            count, relabel, sid = readVol(snc)
            for z in range(0,chunk_sz[0],zstep):
                zz = self.im_id[z + zchunk]
                snd = self.out_seg2d_db % zz + '.h5' 
                if not os.path.exists(snd):
                    print(zz)
                    sn = self.out_seg2d % (zz)
                    seg = readVol(sn, 'main')
                    if do_soma:
                        soma_rl = readVol(sn, 'soma_rl')
                        rl = np.arange(soma_rl.max()+1).astype(np.uint32)
                        rl[soma_rl[:,1]] = count[-1] + soma_rl[:,0]
                        seg = rl[seg]
                    seg[(seg>0)*(seg<=count[-1])] = relabel[seg[(seg>0)*(seg<=count[-1])]+count[z]]
                    writeH5(snd, seg)
 
    def segChunkToAll(self):
        rg_m1_aff = param['RG_MERGE_AFF']
        sna = self.out_seg2d_all
        if not os.path.exists(sna):
            cc = np.zeros(1+(self.ran[0][1]-self.ran[0][0])//self.chunk_sz[0])
            for zi,zchunk in enumerate(range(self.ran[0][0],self.ran[0][1],self.chunk_sz[0])):
                snc = self.out_seg2d_chunk % zchunk 
                cc[zi+1] = readVol(snc, 'sid')[0]
            cc = np.cumsum(cc)

            matches = np.zeros([0,2])
            soma_m = 0
            for zi,zchunk in enumerate(range(self.ran[0][0], self.ran[0][1]-self.chunk_sz[0], self.chunk_sz[0])):
                sno = self.out_seg2d_rgz % (zchunk)
                rg_id, rg_score = readVol(sno, ['id', 'score'])

                sn = self.out_seg2d % (zchunk+self.chunk_sz[0]-1)
                soma_rl = readVol(sn, 'soma_rl')
                rl = np.arange(soma_rl.max()+1).astype(np.uint32)
                rl[soma_rl[:,1]] = cc[-1] + soma_rl[:,0]
                snc = self.out_seg2d_chunk % zchunk 
                count, relabel = readVol(snc, ['count','relabel'])
                gid = rg_id[:,0] < soma_rl[:,1].min()
                rg_id[gid,0] = relabel[rg_id[gid,0]+count[-1]] + cc[zi]
                rg_id[gid==0, 0] = rl[rg_id[gid==0,0]]
                soma_m = max(soma_m, readVol(snc, 'sid')[1])

                sn = self.out_seg2d % (zchunk+self.chunk_sz[0])
                soma_rl = readVol(sn, 'soma_rl')
                rl = np.arange(soma_rl.max()+1).astype(np.uint32)
                rl[soma_rl[:,1]] = cc[-1] + soma_rl[:,0]
                snc = self.out_seg2d_chunk % (zchunk+self.chunk_sz[0]) 
                count, relabel = readVol(snc, ['count','relabel'])
                gid = rg_id[:,1] < soma_rl[:,1].min()
                rg_id[gid,1] = relabel[rg_id[gid,1]+count[0]] + cc[zi+1]
                rg_id[gid==0, 1] = rl[rg_id[gid==0, 1]]
                soma_m = max(soma_m, readVol(snc, 'sid')[1])

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
                bb1 = get_bb_all2d(seg1, uid=ui)
                ui0 = ui
            else:
                bb1 = get_bb_all2d(seg1, uid=ui0)
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
