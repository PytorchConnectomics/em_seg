import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage.morphology import binary_opening, binary_fill_holes, binary_erosion

from skimage.morphology import remove_small_objects, dilation, remove_small_holes
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed

from emu.io import get_kwarg, seg_biggest_cc, seg_to_cc


def get_seg_kwargs(option, **kwargs):
    if option == "binary_post":
        kw = ["num_open", "num_close", "T_small", "fill_holes"]
    elif option == "zwatershed":
        kw = [
            "T_thres",
            "T_dust",
            "T_dust_merge",
            "T_mst_merge",
            "T_low",
            "T_high",
            "T_rel",
        ]

    return [get_kwarg(kwargs, x) for x in kw]


## 1. foreground segmentation
def pred_to_binary(pred, threshold=None):
    # convert foreground probability into binary segment
    if threshold is None:
        threshold = threshold_otsu(pred[pred > 0])
    return pred >= threshold


def binary_postprocessing(
    seg, num_open=None, num_close=None, T_small=None, fill_holes=None
):
    if num_open is not None:
        seg = binary_opening(seg, iterations=num_open)
    if num_close is not None:
        seg = binary_opening(seg, iterations=num_close)

    if T_small is not None:
        if T_small > 0:
            seg = remove_small_objects(seg, T_small)
        else:
            # if T_small is negative, keep the largest
            seg = seg_biggest_cc(seg)
    if fill_holes is not None:
        seg = binary_fill_holes(seg)
    return seg


## 2. instance segmentation
def pred_watershed(pred, T_marker=0.5, peak_size=None, T_fg=None):
    # https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html
    if peak_size is None:
        peak_size = [11, 11]
    marker_mask = pred_to_binary(pred, T_marker)
    distance = ndi.distance_transform_cdt(marker_mask)
    marker_maxima = peak_local_max(
        distance, indices=False, footprint=np.ones((peak_size, peak_size)), labels=image
    )
    markers = seg_to_cc(marker_maxima)
    marker_fg = marker_mask if T_fg is None else pred >= T_marker
    return watershed(-distance, markers, mask=marker_fg)


def boundary_watershed(boundary, peak_size=None, seg_fg=None):
    if peak_size is None:
        peak_size = [11, 11]
    distance = ndi.distance_transform_cdt(boundary)
    maxima = peak_local_max(
        distance, indices=False, footprint=np.ones([peak_size, peak_size])
    )
    markers = seg_to_cc(maxima)
    return watershed(-distance, markers)


def aff_zwatershed(
    aff,
    T_thres=800,
    T_dust=50,
    T_dust_merge=0.2,
    T_mst_merge=0.7,
    T_low=0.1,
    T_high=0.8,
    T_rel=True,
):
    import zwatershed

    T_thres = 800 if T_thres is None else T_thres
    T_dust = 50 if T_dust is None else T_dust
    T_dust_merge = 0.2 if T_dust_merge is None else T_dust_merge
    T_mst_merge = 800 if T_mst_merge is None else T_mst_merge
    T_low = 0.1 if T_low is None else T_low
    T_high = 0.8 if T_high is None else T_high
    T_rel = True if T_rel is None else T_rel

    if aff.dtype == np.uint8:
        aff = aff.astype(np.float32) / 255
    return zwatershed.zwatershed(
        aff,
        T_threshes=[T_thres],
        T_dust=T_dust,
        T_aff=[T_low, T_high, T_dust_merge],
        T_aff_relative=T_rel,
        T_merge=T_mst_merge,
    )[0][0][0]


def probToInstanceSeg_cc(probability, thres_prob=0.8, thres_small=128):
    # connected component
    if probability.max() > 1:
        probability = probability / 255.0
    foreground = probability > thres_prob
    segm = label(foreground)
    segm = remove_small_objects(segm, thres_small)
    return segm.astype(np.uint32)


def probToInstanceSeg_watershed(probability, thres1=0.98, thres2=0.85, do_resize=False):
    # watersehd
    seed_map = probability > 255 * thres1
    foreground = probability > 255 * thres2
    seed = label(seed_map)
    segm = watershed(-semantic, seed, mask=foreground)
    segm = remove_small_objects(segm, 128)
    if do_resize:
        target_size = (
            semantic.shape[0],
            semantic.shape[1] // 2,
            semantic.shape[2] // 2,
        )
        segm = resize(
            segm, target_size, order=0, anti_aliasing=False, preserve_range=True
        )
    return segm.astype(np.uint32)


def boundaryContourToSeg(volume, thres1=0.8, thres2=0.4, do_resize=False):
    semantic = volume[0]
    boundary = volume[1]
    foreground = (semantic > int(255 * thres1)) * (boundary < int(255 * thres2))
    # foreground = (semantic > int(255*thres1))
    segm = label(foreground)
    struct = np.ones((1, 5, 5))
    segm = dilation(segm, struct)
    segm = remove_small_objects(segm, 128)
    if do_resize:
        target_size = (
            semantic.shape[0],
            semantic.shape[1] // 2,
            semantic.shape[2] // 2,
        )
        segm = resize(
            segm, target_size, order=0, anti_aliasing=False, preserve_range=True
        )
    return segm.astype(np.uint32)


def probBoundaryToSeg(volume, thres1=0.9, thres2=0.8, thres3=0.85, do_resize=False):
    semantic = volume[0]
    boundary = volume[1]
    seed_map = (semantic > int(255 * thres1)) * (boundary < int(255 * thres2))
    foreground = semantic > int(255 * thres3)
    seed = label(seed_map)
    segm = watershed(-semantic, seed, mask=foreground)
    segm = remove_small_objects(segm, 128)
    if do_resize:
        target_size = (
            semantic.shape[0],
            semantic.shape[1] // 2,
            semantic.shape[2] // 2,
        )
        segm = resize(
            segm, target_size, order=0, anti_aliasing=False, preserve_range=True
        )
    return segm.astype(np.uint32)


def watershed_3d(volume):
    # https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html
    distance = ndi.distance_transform_edt(volume)
    local_maxi = peak_local_max(
        distance, indices=False, footprint=np.ones((3, 3, 3)), labels=volume
    )
    markers = ndi.label(local_maxi)[0]
    seg = watershed(-distance, markers, mask=volume)
    return seg


def watershed_3d_open(volume, res=[1, 1, 1], erosion_iter=0, marker_size=0, zcut=-1):
    markers = ndi.label(binary_erosion(volume, iterations=erosion_iter))[0]
    if marker_size > 0:
        ui, uc = np.unique(markers, return_counts=True)
        rl = np.arange(1 + ui.max()).astype(volume.dtype)
        rl[ui[uc < marker_size]] = 0
        markers = rl[markers]
    marker_id = np.unique(markers)
    if (marker_id > 0).sum() == 1:
        # try the simple z-2D cut
        if zcut > 0 and volume.shape[0] >= zcut * 1.5:
            from scipy.signal import find_peaks

            zsum = (volume > 0).sum(axis=1).sum(axis=1)
            peaks = find_peaks(zsum, zsum.max() * 0.7, distance=zcut * 0.8)[0]
            if len(peaks) > 1:
                markers[:] = 0
                for i, peak in enumerate(peaks):
                    markers[peak][volume[peak] > 0] = 1 + i
                marker_id = np.arange(len(peaks) + 1)

    if (marker_id > 0).sum() == 1:
        seg = volume
    else:
        if min(res) == max(res):
            distance = ndi.distance_transform_edt(volume)
        else:
            import edt

            # zyx: order='C'
            distance = edt.edt(volume, anisotropy=res)
        seg = watershed(-distance, markers, mask=volume)
    return seg


def imToSeg_2d(im, th_hole=512, T_small=10000, seed_footprint=[71, 71]):
    if len(np.unique(im)) == 1:
        return (im < 0).astype(np.uint8)
    thresh = threshold_otsu(im)
    seg = remove_small_objects(
        binary_fill_holes(remove_small_holes(im > thresh, th_hole)), T_small
    )
    if seed_footprint[0] > 0:
        seg = watershed_2d(seg, seed_footprint)
    return seg


def watershed_2d(volume, seed_footprint=[71, 71]):
    distance = ndi.distance_transform_edt(volume)
    local_maxi = peak_local_max(
        distance, indices=False, footprint=np.ones(seed_footprint), labels=volume
    )
    seg_cc = label(volume)
    bid = list(set(range(1, seg_cc.max() + 1)) - set(np.unique(seg_cc[local_maxi])))
    # fill missing seg
    for i in bid:
        tmp = ndi.binary_erosion(seg_cc == i, iterations=10)
        local_maxi[tmp > 0] = 1

    markers = ndi.label(local_maxi)[0]
    seg = watershed(-distance, markers, mask=volume)
    return seg
