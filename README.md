# Copytyping across modalities
Given copy-number profile (CNP) from matched bulk sequencing data and phasing information, copytyping can assign cells/spots to individual clones defined by CNP using both total feature and allelic feature.

## Support modality
1. scRNA-seq
2. scATAC-seq
3. scMultiome
4. Visium

## Modules
1. `src/build_tile_matrix_atac.py`
2. `src/annotate_anndata.py`
3. `src/preprocess.py`
4. `src/copytyping.py`

## Dependencies
Basic python libraries like Numpy, Pandas, SciPy, etc., are required

### Preprocessing
1. pyranges
2. scanpy
3. squidpy
3. snapatac2

### Copytyping
1. pytorch
2. jax
3. jaxopt
4. statsmodels
