import sys
import glob
from emu.io import split_arr_by_chunk, mkdir, read_vol, write_h5
from seglib import vol_to_iou_parallel, iou_to_matches_parallel, merge_id

def test_seg_track(seg, iou_thres=0.2):
    from imu.seg import predToSeg2d,seg2dToIoU
    seg3d_naive = predToSeg2d(seg, [-1])
    del seg
    matches = seg2dToIoU(seg3d_naive, iou_thres)
    seg3d = seg2dTo3d(seg3d_naive, matches)
    return seg3d


if __name__ == "__main__":
    opt = sys.argv[1]
    job_id, job_num = 0, 1
    if len(sys.argv) > 3:
        job_id, job_num = int(sys.argv[2]), int(sys.argv[3])
    result_seg2d = '/mmfs1/data/wanjr/data/p3/%04d.h5'
    index = sorted([int(x[x.rfind('/')+1:-3]) for x in glob.glob(os.path.dirname(result_seg2d))])
    output_folder = 'temp/track_seg2d/'
    mkdir(output_folder)
    
    th_matches = 0.3
    output_iou = f'{output_folder}/iou_%d_%d.npy'
    output_global_id = f'{output_folder}/global_id.h5'

    index_parallel = split_arr_by_chunk(index, job_id, job_num, 1)
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
        write_slurm_all('source activate imu\n python /data/weidf/lib/seglib/demo/track_seg2d.py 0 %d %d \n', os.path.join(output_folder, 'iou'), job_num)
    elif opt == '1':
        # compute seg2d global id 
        if not os.path.exists(output_global_id):
            global_id = iou_list_to_global_id(iou_list)
            write_h5(output_global_id, global_id)
    elif opt == '2':
        # compute all matches and seg id relabel
        # compute all matches
        global_id = read_vol(output_global_id)
        matches = iou_list_to_matches(iou_list, th_iou, global_id).astype(np.uint32)
        # compute seg id relabel
        seg_relabel = merge_id(matches[0], matches[1])
        write_h5(output_seg_relabel, seg_relabel)
    elif opt == '3':
        # output relabeld slice
        if job_id != job_num-1:
            # remove the overlapping image
            index_parallel = index_parallel[:-1]
        for i in index_parallel:
            seg = get_seg 
    elif opt == '3.1':
        # write slurm files
        write_slurm_all('source activate imu\n python /data/weidf/lib/seglib/demo/track_seg2d.py 3 %d %d \n', os.path.join(output_folder, 'iou'), job_num)
    elif opt == '4':
        # low-res visualization

