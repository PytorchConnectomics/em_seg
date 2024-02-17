Recipe for SNEMI test result

1. Affinity from Unet
    - replace z=29 with z=28 due to the alignment error
    - sbatch test.sh

2. Segmentation from affinity
    - init 2d seg: zwatershed
    - 3D seg: waterz agglomeration
    - post-processing

