import os,sys
import numpy as np

def mkdir(path, opt=''):
    if opt == 'parent':
        path = os.path.dirname(path)
    if not os.path.exists(path):
        os.makedirs(path)

def removeId(arr1, arr2, invert=True):
    # invert=True: remove intersection
    # invert=False: intersection of two arrays 
    return arr1[np.in1d(arr1, arr2, invert=invert)]


def writeh5(filename, dtarray, datasetname='main'):
    import h5py
    fid=h5py.File(filename,'w')
    if isinstance(datasetname, (list,)):
        for i,dd in enumerate(datasetname):
            ds = fid.create_dataset(dd, dtarray[i].shape, compression="gzip", dtype=dtarray[i].dtype)
            ds[:] = dtarray[i]
    else:
        ds = fid.create_dataset(datasetname, dtarray.shape, compression="gzip", dtype=dtarray.dtype)
        ds[:] = dtarray
    fid.close()

def readh5(filename, datasetname=None):
    import h5py
    fid = h5py.File(filename,'r')

    if datasetname is None:
        if sys.version[0]=='2': # py2
            datasetname = fid.keys()
        else: # py3
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
