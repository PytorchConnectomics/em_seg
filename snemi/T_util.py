import numpy as np
import h5py
import mahotas
import scipy.sparse as sparse
from scipy.ndimage.morphology import binary_fill_holes
def relabel(seg, do_dtype = False):
    if seg is None or seg.max()==0:
        return seg
    uid = np.unique(seg)
    uid = uid[uid > 0]
    max_id = int(max(uid))
    mapping = np.zeros(max_id + 1, dtype = seg.dtype)
    mapping[uid] = np.arange(1, len(uid) + 1)
    if do_dtype:
        return relabelDtype(mapping[seg])
    else:
        return mapping[seg]

def relabelDtype(seg):
    max_id = seg.max()
    m_type = np.uint64
    if max_id<2**8:
        m_type = np.uint8
    elif max_id<2**16:
        m_type = np.uint16
    elif max_id<2**32:
        m_type = np.uint32
    return seg.astype(m_type)


def seg_postprocess(seg, sids=[]):
    # watershed fill the unlabeled part
    if seg.ndim == 3:
        for z in range(seg.shape[0]):
            seg[z] = mahotas.cwatershed(seg[z]==0, seg[z])
            for sid in sids:
                tmp = binary_fill_holes(seg[z]==sid)
                seg[z][tmp>0] = sid
    elif seg.ndim == 2:
        seg = mahotas.cwatershed(seg==0, seg)
    return seg


def readh5(filename, datasetname=None):
    fid = h5py.File(filename,'r')

    if datasetname is None:
        datasetname = list(fid)
    if len(datasetname) == 1:
        datasetname = datasetname[0]
    if isinstance(datasetname, (list,)):
        out=[None]*len(datasetname)
        for di,d in enumerate(datasetname):
            out[di] = np.array(fid[d])
        return out
    else:
        return np.array(fid[datasetname])

def writeh5(filename, dtarray, datasetname='main'):
    fid=h5py.File(filename,'w')
    if isinstance(datasetname, (list,)):
        for i,dd in enumerate(datasetname):
            ds = fid.create_dataset(dd, dtarray[i].shape, compression="gzip", dtype=dtarray[i].dtype)
            ds[:] = dtarray[i]
    else:
        ds = fid.create_dataset(datasetname, dtarray.shape, compression="gzip", dtype=dtarray.dtype)
        ds[:] = dtarray
    fid.close()


def adapted_rand(seg, gt, all_stats=False):
    """Compute Adapted Rand error as defined by the SNEMI3D contest [1]
    Formula is given as 1 - the maximal F-score of the Rand index 
    (excluding the zero component of the original labels). Adapted 
    from the SNEMI3D MATLAB script, hence the strange style.
    Parameters
    ----------
    seg : np.ndarray
        the segmentation to score, where each value is the label at that point
    gt : np.ndarray, same shape as seg
        the groundtruth to score against, where each value is a label
    all_stats : boolean, optional
        whether to also return precision and recall as a 3-tuple with rand_error
    Returns
    -------
    are : float
        The adapted Rand error; equal to $1 - \frac{2pr}{p + r}$,
        where $p$ and $r$ are the precision and recall described below.
    prec : float, optional
        The adapted Rand precision. (Only returned when `all_stats` is ``True``.)
    rec : float, optional
        The adapted Rand recall.  (Only returned when `all_stats` is ``True``.)
    References
    ----------
    [1]: http://brainiac2.mit.edu/SNEMI3D/evaluation
    """
    # segA is truth, segB is query
    segA = np.ravel(gt)
    segB = np.ravel(seg)
    n = segA.size

    n_labels_A = np.amax(segA) + 1
    n_labels_B = np.amax(segB) + 1

    ones_data = np.ones(n,int)

    p_ij = sparse.csr_matrix((ones_data, (segA[:], segB[:])), shape=(n_labels_A, n_labels_B))

    a = p_ij[1:n_labels_A,:]
    b = p_ij[1:n_labels_A,1:n_labels_B]
    c = p_ij[1:n_labels_A,0].todense()
    d = b.multiply(b)

    a_i = np.array(a.sum(1))
    b_i = np.array(b.sum(0))

    sumA = np.sum(a_i * a_i)
    sumB = np.sum(b_i * b_i) + (np.sum(c) / n)
    sumAB = np.sum(d) + (np.sum(c) / n)

    precision = sumAB / float(sumB)
    recall = sumAB / float(sumA)

    fScore = 2.0 * precision * recall / (precision + recall)
    are = 1.0 - fScore

    if all_stats:
        return (are, precision, recall)
    else:
        return are


