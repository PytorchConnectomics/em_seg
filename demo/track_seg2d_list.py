import os
import sys
import glob
import numpy as np
from emu.io import split_arr_by_chunk, mkdir, read_vol, write_h5, seg_list_to_remapped
from emu.cluster import write_slurm_all
from seglib import seg_list_to_iou, iou_list_to_global_id, iou_list_to_matches, merge_id


if __name__ == "__main__":
    result_seg2d = '/mmfs1/data/wanjr/data/p3/%04d.h5'
    th_iou = 0.3
    vis_ratio = [8,4,4]
    file_path = '/data/weidf/lib/seglib/demo/track_seg2d.py'
    output_folder = 'temp/track_seg2d/'
    output_iou = f'{output_folder}/iou_%d_%d.npy'
    output_global_id = f'{output_folder}/global_id_{th_iou}.h5'
    output_seg_mapping = f'{output_folder}/relabel_{th_iou}.h5' 
    output_seg = f'{output_folder}/seg_{th_iou}/%d.h5' 
    output_vis = f'{output_folder}/seg_{th_iou}_{vis_ratio[0]}_{vis_ratio[1]}_{vis_ratio[2]}.h5' 


    opt = sys.argv[1]
    job_id, job_num = 0, 1
    if len(sys.argv) > 3:
        job_id, job_num = int(sys.argv[2]), int(sys.argv[3])
    index = sorted([int(x[x.rfind('/')+1:-3]) for x in glob.glob(os.path.dirname(result_seg2d))])
    index_parallel = split_arr_by_chunk(index, job_id, job_num, 1)
    mkdir(output_folder)
    seg_list = [result_seg2d % z for z in index_parallel] 
    iou_list = [output_iou % (x, job_num) for x in range(job_num)]
    if opt == '0':
        # compute matches in the parallel fashion
        sn = output_iou % (job_id, job_num)
        if not os.path.exists(sn):
            seg_list = [result_seg2d % z for z in index_parallel]
            out = seg_list_to_iou(seg_list)
            np.save(sn, out)
    elif opt == '0.1':
        # write slurm files
        write_slurm_all(f'source activate imu\n python {file_path} 0 %d %d \n', os.path.join(output_folder, 'iou'), job_num)
    elif opt == '1':
        # single machine
        # compute global id
        if not os.path.exists(output_global_id):
            global_id = iou_list_to_global_id(iou_list)
            write_h5(output_global_id, global_id)            
        # compute all matches
        global_id = read_vol(output_global_id)
        matches = iou_list_to_matches(iou_list, th_iou, global_id).astype(np.uint32)
        # compute seg id relabel
        mapping = merge_id(matches[0], matches[1])
        write_h5(output_seg_mapping, mapping)
    elif opt == '2':
        # output relabeld slice
        global_id = read_vol(output_global_id)
        if job_id != job_num-1:
            # remove the overlapping image
            seg_list = seg_list[:-1]
            output_list = output_list[:-1]
            global_id = global_id[len(seg_list)*job_id: len(seg_list)*(job_id+1)]
        else:
            global_id = global_id[-len(seg_list):]
        mapping = read_vol(output_seg_mapping)
        seg_list_to_remapped(seg_list, mapping, global_id, output_list)
    elif opt == '2.1':
        # write slurm files
        write_slurm_all(f'source activate imu\n python {file_path} 2 %d %d \n', os.path.join(output_folder, 'relabel'), job_num)
    elif opt == '3':
        # low-res visualization
        index_vis = index[::vis_ratio[0]]
        seg = read_vol(output_seg % index_vis[0])[::vis_ratio[1], ::vis_ratio[2]]
        out = np.zeros([len(index_vis)] + list(seg.shape), dtype=seg.dtype)
        out[0] = seg
        for i, ind in enumerate(index_vis[1:]):
            out[i + 1] = read_vol(output_seg % ind)[::vis_ratio[1], ::vis_ratio[2]]
        write_h5(output_vis, out)
        