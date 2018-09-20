"""
Implementation of MEV-SFS algorithm

Detail can be found in paper "A Geometry-Based Band Selection
Approach for Hyperspectral Image Analysis" in Algorithm 1
"""

from osgeo import gdal
import numpy as np
from tqdm_progress import TqdmUpTo
import scipy.io as scio
import matplotlib.pyplot as plt


def load_gdal_data(data_path):
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
    return databuf_in.transpose()


def mev_sfs(data: np.ndarray, sel_band_count, removed_bands=None):
    bands = data.shape[1]
    band_idx_map = np.arange(bands)

    if not (removed_bands is None):
        data = np.delete(data, removed_bands, 1)
        bands = bands - len(removed_bands)
        band_idx_map = np.delete(band_idx_map, removed_bands)

    data_mean = np.mean(data, axis=0)
    data = data - data_mean
    data_var = np.var(data, axis=0)
    cov_mat = np.matmul(data.transpose(), data) / (data.shape[0])

    sel_bands = np.array([np.argmax(data_var)])
    current_selected_count = 1
    while current_selected_count < sel_band_count:
        max_mev = 0
        new_sel_band = -1
        for i in range(bands):
            if not (i in sel_bands):
                new_sel_bands = np.append(sel_bands, i)
                a = np.ix_(new_sel_bands, new_sel_bands)
                new_cov_mat = cov_mat[np.ix_(new_sel_bands, new_sel_bands)]
                mev = np.linalg.det(new_cov_mat)
                if mev > max_mev:
                    max_mev = mev
                    new_sel_band = i
        sel_bands = np.append(sel_bands, new_sel_band)
        current_selected_count += 1

    print(band_idx_map[sel_bands] + 1)
    print(np.sort(band_idx_map[sel_bands] + 1))


def main():
    # mev_sfs(r"I:\hyperspec-nano \20180828\6_2018_08_28_10_57_19\raw_T_nodark", 5)
    remove_bands = [0, 1, 2, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                    111, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156,
                    157, 158, 159, 160, 161, 162, 163, 164, 216, 217, 218,
                    219]

    use_mat_data = False

    if use_mat_data:
        mat_dict = scio.loadmat(r"F:\Data\Benchmark\Hyperspec\Indian_pines.mat")

        for key in mat_dict:
            if type(mat_dict[key]) is np.ndarray:
                image_data = mat_dict[key]  # type: np.ndarray

        cols = image_data.shape[1]
        rows = image_data.shape[0]
        bands = image_data.shape[2]

        image_data = image_data.reshape(cols * rows, bands)
        mev_sfs(image_data, 15, remove_bands)
    else:
        image_data = load_gdal_data(r"E:\Download\10_4231_R7RX991C\aviris_hyperspectral_data"
                                    r"\19920612_AVIRIS_IndianPine_Site3.tif")
        mev_sfs(image_data, 15, remove_bands)


if __name__ == "__main__":
    main()
