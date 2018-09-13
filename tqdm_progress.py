from tqdm import tqdm


class TqdmUpTo(tqdm):
    """
    tqdm helper class to be a callback function for gdal when reading data
    Show console progress

    Provides `update_to(n)` which uses `tqdm.update(delta_n)`.
    """
    def update_to(self, complete, message, callback_data):
        self.update(int(complete * 100) - self.n)

