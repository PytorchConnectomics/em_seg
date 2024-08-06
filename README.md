# em_seg


```
conda create -n em_seg python==3.7
source activate em_seg


git clone git@github.com:PytorchConnectomics/em_util.git
cd em_util
pip install --editable .
cd ..


git clone git@github.com:PytorchConnectomics/em_seg.git
cd em_seg
pip install --editable .
# optional for c++ package
python setup.py build_ext --inplace
```
