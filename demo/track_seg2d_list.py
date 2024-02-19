import os
import sys
import glob
import numpy as np
from tqdm import tqdm
from emu.io import split_arr_by_chunk, mkdir, read_vol, write_h5, seg_to_cc
from emu.cluster import write_slurm_all
from seglib import (
    seg_list_to_iou2d,
    seg_list_to_iou,
    iou_list_to_global_id,
    iou_list_to_matches,
    seg_list_to_remapped,
    merge_id,
)


if __name__ == "__main__":
    file_path = "/data/weidf/lib/seglib/demo/track_seg2d_list.py"
    result_seg2d = "/mmfs1/data/bccv/dataset/nagP3/jia_nucleus/*.h5"
    th_iou = 0.3
    vis_ratio = [4, 2, 2]

    output_folder = os.path.join(os.path.dirname(file_path), "temp/track_seg2d/")
    output_iou = f"{output_folder}/iou_%d_%d.h5"
    output_global_id = f"{output_folder}/global_id_{th_iou}.h5"
    output_seg_mapping = f"{output_folder}/relabel_{th_iou}.h5"
    output_seg = f"{output_folder}/seg_{th_iou}/%d.h5"
    output_vis = (
        f"{output_folder}/seg_{th_iou}_{vis_ratio[0]}_{vis_ratio[1]}_{vis_ratio[2]}.h5"
    )

    opt = sys.argv[1]
    job_id, job_num = 0, 1
    if len(sys.argv) > 3:
        job_id, job_num = int(sys.argv[2]), int(sys.argv[3])
    seg_list_all = sorted(glob.glob(result_seg2d))
    seg_name_all = [int(os.path.basename(x).split(".")[0]) for x in seg_list_all]
    seg_index_all = range(len(seg_list_all))

    # for each job_id
    seg_index = split_arr_by_chunk(seg_index_all, job_id, job_num, 1)
    mkdir(output_folder)
    seg_list = [seg_list_all[x] for x in seg_index]
    iou_list = [output_iou % (x, job_num) for x in range(job_num)]
    if opt == "0":
        # compute matches in the parallel fashion
        sn = output_iou % (job_id, job_num)
        if not os.path.exists(sn):
            out = seg_list_to_iou2d(seg_list, add_last=job_id == job_num - 1)
            write_h5(sn, out)            
    elif opt == "0.1":
        # write slurm files
        write_slurm_all(
            f"source /data/weidf/miniconda3/bin/activate emu\npython {file_path} 0 %d %d \n",
            os.path.join(output_folder, "iou"),
            job_num, memory=10000
        )
    elif opt == "1":
        # single machine
        # compute global id
        if not os.path.exists(output_global_id):
            global_id = iou_list_to_global_id(iou_list)
            write_h5(output_global_id, global_id)
        else:
            global_id = read_vol(output_global_id)
        # compute all matches
        matches = iou_list_to_matches(iou_list, th_iou, global_id).astype(np.uint32)
        # compute seg id relabel
        mapping = merge_id(matches[:, 0], matches[:, 1])
        write_h5(output_seg_mapping, mapping)
    elif opt == "2":
        # output relabeled segments
        global_id = read_vol(output_global_id)
        if job_id != job_num - 1:
            # remove the overlapping seg
            seg_index = seg_index[:-1]
            seg_list = seg_list[:-1]
            global_id = global_id[len(seg_list) * job_id : len(seg_list) * (job_id + 1)]
        else:
            global_id = global_id[-len(seg_list) :]
        output_list = [output_seg % x for x in seg_index]
        mapping = read_vol(output_seg_mapping)
        seg_list_to_remapped(seg_list, mapping, global_id, output_list)
    elif opt == "2.1":
        # write slurm files
        write_slurm_all(
            f"source activate emu\n python {file_path} 2 %d %d \n",
            os.path.join(output_folder, "relabel"),
            job_num,
        )
    elif opt == "3":
        # low-res visualization
        index_vis = index[:: vis_ratio[0]]
        seg = read_vol(output_seg % index_vis[0])[:: vis_ratio[1], :: vis_ratio[2]]
        out = np.zeros([len(index_vis)] + list(seg.shape), dtype=seg.dtype)
        out[0] = seg
        for i, ind in tqdm(enumerate(index_vis[1:])):
            out[i + 1] = read_vol(output_seg % ind)[:: vis_ratio[1], :: vis_ratio[2]]
        write_h5(output_vis, out)
    elif opt == "3.1":
        # debug
        iou_list = [output_iou % (0, job_num)]
        gid = iou_list_to_global_id(iou_list)        
        matches = iou_list_to_matches(iou_list, th_iou, gid).astype(np.uint32)
        # compute seg id relabel
        from seglib.seg_track import seg3d_to_remapped
        mapping = merge_id(matches[:,0], matches[:,1])
        seg = read_vol(seg_list_all[0])[:, :: vis_ratio[1], :: vis_ratio[2]]
        out = seg3d_to_remapped(seg.copy(), mapping, gid)                    
        import neuroglancer
        ip='localhost' # or public IP of the machine for sharable display
        port=9092 # change to an unused port number
        neuroglancer.set_server_bind_address(bind_address=ip, bind_port=port)
        viewer=neuroglancer.Viewer()
        from emu.ng import ng_layer
        # set volume resolution: order xyz
        res= [128,128,120]
        fn_im = 'precomputed://https://rhoana.rc.fas.harvard.edu/ng/nag_p3/im/'
        print(np.unique(out))
        with viewer.txn() as s:
            #s.layers['image'] = neuroglancer.ImageLayer(source=fn_im)
            s.layers.append(name='seg', layer=ng_layer(seg, res))
            s.layers.append(name='out', layer=ng_layer(out, res))
        print(viewer)
