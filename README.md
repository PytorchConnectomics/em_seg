# seglib


```
conda create -n seglib python==3.7
source activate seglib


git clone git@github.com:PytorchConnectomics/EM-util.git
cd EM-util
pip install --editable .
cd ..


git clone git@github.com:PytorchConnectomics/seglib.git
cd seglib
pip install --editable .
# optional for c++ package
python setup.py build_ext --inplace
```
