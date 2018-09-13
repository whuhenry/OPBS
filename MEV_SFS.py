"""
Implementation of MEV-SFS algorithm

Detail can be found in paper "A Geometry-Based Band Selection
Approach for Hyperspectral Image Analysis" in Algorithm 1
"""

from osgeo import gdal
import numpy as np
from tqdm_progress import TqdmUpTo


def mev_sfs(data_path, sel_band_count):
    dataset: gdal.Dataset = gdal.Open(data_path, gdal.GA_ReadOnly)
    if not dataset:
        return None

    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    bands = dataset.RasterCount

    with TqdmUpTo(unit='%', unit_scale=False, miniters=1, total=100,
                  desc="reading file %s" % data_path) as t:
        databuf_in: np.ndarray = dataset.ReadAsArray(callback=t.update_to)
    databuf_in = databuf_in.reshape(bands, rows * cols)

    # Compute covariance and variance for each band
    data_mean = np.mean(databuf_in, axis=1)
    data_var = np.var(databuf_in, axis=1)
    databuf_in = databuf_in.astype(np.float64).transpose() - data_mean
    cov_mat = databuf_in.transpose().dot(databuf_in) / (cols * rows)

    sel_bands = np.array([np.argmax(data_var)])
    current_selected_count = 1
    while current_selected_count < sel_band_count:
        max_mev = 0
        new_sel_band = -1
        for i in range(bands):
            if not (i in sel_bands):
                new_sel_bands = np.append(sel_bands, i)
                new_cov_mat = cov_mat[np.ix_(new_sel_bands, new_sel_bands)]
                mev = np.linalg.det(np.dot(new_cov_mat.transpose(), new_cov_mat))
                if mev > max_mev:
                    max_mev = mev
                    new_sel_band = i
        sel_bands = np.append(sel_bands, new_sel_band)
        current_selected_count += 1

    print(np.sort(sel_bands))

    return sel_bands


if __name__ == "__main__":
    mev_sfs(r"I:\hyperspec-nano\20180828\6_2018_08_28_10_57_19\raw_T_nodark", 5)
