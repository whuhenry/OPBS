"""
Implementation of OPBS algorithm

Detail can be found in paper "A Geometry-Based Band Selection
Approach for Hyperspectral Image Analysis" in Algorithm 2
"""


import numpy as np
from MEV_SFS import load_gdal_data
import scipy.io as scio


def opbs(image_data, sel_band_count, removed_bands=None):
    if image_data is None:
        return None

    bands = image_data.shape[1]
    band_idx_map = np.arange(bands)

    if not (removed_bands is None):
        image_data = np.delete(image_data, removed_bands, 1)
        bands = bands - len(removed_bands)
        band_idx_map = np.delete(band_idx_map, removed_bands)

    # Compute covariance and variance for each band
    # TODO: data normalization to all band
    data_mean = np.mean(image_data, axis=0)
    image_data = image_data - data_mean
    data_var = np.var(image_data, axis=0)
    h = data_var * image_data.shape[0]
    op_y = image_data.transpose()

    sel_bands = np.array([np.argmax(data_var)])
    last_sel_band = sel_bands[0]
    current_selected_count = 1
    while current_selected_count < sel_band_count:
        for t in range(bands):
            if not (t in sel_bands):
                op_y[t] = op_y[t] - np.dot(op_y[last_sel_band], op_y[t]) / h[last_sel_band] * op_y[last_sel_band]

        max_h = 0
        new_sel_band = -1
        for t in range(bands):
            if not (t in sel_bands):
                h[t] = np.dot(op_y[t], op_y[t])
                if h[t] > max_h:
                    max_h = h[t]
                    new_sel_band = t
        sel_bands = np.append(sel_bands, new_sel_band)
        print(max_h)
        current_selected_count += 1

    print(sel_bands + 1)
    print(band_idx_map[sel_bands] + 1)

    return sel_bands


def main():
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
    else:
        image_data = load_gdal_data(r"E:\Download\10_4231_R7RX991C\aviris_hyperspectral_data"
                                    r"\19920612_AVIRIS_IndianPine_Site3.tif")

    opbs(image_data, 15, remove_bands)


if __name__ == "__main__":
    main()
