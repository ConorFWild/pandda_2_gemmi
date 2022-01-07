from __future__ import annotations

import typing
import dataclasses
from pathlib import Path
from functools import partial

from scipy import stats
from joblib.externals.loky import set_loky_pickler
set_loky_pickler('pickle')
from scipy import optimize
import seaborn as sns
sns.set_theme()

from pandda_gemmi.constants import *
from pandda_gemmi.common import Dtag
from pandda_gemmi.shells import Shell
from pandda_gemmi.edalignment import XmapArray, Xmap, Grid, Xmaps
from pandda_gemmi.python_types import *


@dataclasses.dataclass()
class Model:
    mean: np.array
    sigma_is: typing.Dict[Dtag, float]
    sigma_s_m: np.ndarray

    @staticmethod
    def mean_from_xmap_array(masked_train_xmap_array: XmapArray):
        mean_flat = np.mean(masked_train_xmap_array.xmap_array, axis=0)

        return mean_flat

    @staticmethod
    def sigma_is_from_xmap_array(masked_train_xmap_array: XmapArray,
                                 mean_array: np.ndarray,
                                 cut: float,
                                 ):
        # Estimate the dataset residual variability
        sigma_is = {}
        for dtag in masked_train_xmap_array:
            sigma_i = Model.calculate_sigma_i(mean_array,
                                              masked_train_xmap_array[dtag],
                                              cut,
                                              )
            sigma_is[dtag] = sigma_i

        return sigma_is

    @staticmethod
    def sigma_sms_from_xmaps(masked_train_xmap_array: XmapArray,
                             mean_array: np.ndarray,
                             sigma_is: typing.Dict[Dtag, float],
                             process_local,
                             ):
        # Estimate the adjusted pointwise variance
        sigma_is_array = np.array([sigma_is[dtag] for dtag in masked_train_xmap_array],
                                  dtype=np.float32)[:, np.newaxis]
        sigma_s_m_flat = Model.calculate_sigma_s_m_np(
            mean_array,
            masked_train_xmap_array.xmap_array,
            sigma_is_array,
            process_local,
        )

        return sigma_s_m_flat

    @staticmethod
    def from_mean_is_sms(mean_flat,
                         sigma_is,
                         sigma_s_m_flat,
                         grid: Grid, ):

        # mask = grid.partitioning.protein_mask
        # mask_array = np.array(mask, copy=False, dtype=np.int8)

        total_mask = grid.partitioning.total_mask

        mean = np.zeros(total_mask.shape, dtype=np.float32)
        mean[total_mask == 1] = mean_flat

        sigma_s_m = np.zeros(total_mask.shape, dtype=np.float32)
        sigma_s_m[total_mask == 1] = sigma_s_m_flat

        return Model(
            mean,
            sigma_is,
            sigma_s_m,
        )

    # @staticmethod
    # def from_xmaps(xmaps: Xmaps, grid: Grid, cut: float):
    #     mask = grid.partitioning.protein_mask
    #     mask_array = np.array(mask, copy=False, dtype=np.int8)
    #
    #     arrays = {}
    #     for dtag in xmaps:
    #         xmap = xmaps[dtag]
    #         xmap_array = xmap.to_array()
    #
    #         arrays[dtag] = xmap_array[np.nonzero(mask_array)]
    #
    #     stacked_arrays = np.stack(list(arrays.values()), axis=0)
    #     mean_flat = np.mean(stacked_arrays, axis=0)
    #
    #     # Estimate the dataset residual variability
    #     sigma_is = {}
    #     for dtag in xmaps:
    #         sigma_i = Model.calculate_sigma_i(mean_flat,
    #                                           arrays[dtag],
    #                                           cut)
    #         sigma_is[dtag] = sigma_i
    #
    #     # Estimate the adjusted pointwise variance
    #     sigma_is_array = np.array(list(sigma_is.values()), dtype=np.float32)[:, np.newaxis]
    #     sigma_s_m_flat = Model.calculate_sigma_s_m(mean_flat,
    #                                                stacked_arrays[:60],
    #                                                sigma_is_array[:60],
    #                                                )
    #
    #     mean = np.zeros(mask_array.shape, dtype=np.float32)
    #     mean[np.nonzero(mask_array)] = mean_flat
    #
    #     sigma_s_m = np.zeros(mask_array.shape, dtype=np.float32)
    #     sigma_s_m[np.nonzero(mask_array)] = sigma_s_m_flat
    #
    #     return Model(mean,
    #                  sigma_is,
    #                  sigma_s_m,
    #                  )

    @staticmethod
    def calculate_sigma_i(mean: np.array, array: np.array, cut: float):
        # TODO: Make sure this is actually equivilent
        # Calculated from slope of array - mean distribution against normal(0,1)
        residual = np.subtract(array, mean)
        observed_quantile_estimates = np.sort(residual)
        # sigma_i = np.std(residual)
        # sigma_i = 0.01

        percentiles = np.linspace(0, 1, array.size + 2)[1:-1]
        normal_quantiles = stats.norm.ppf(percentiles)

        below_min_mask = normal_quantiles < (-1.0 * cut)
        above_max_mask = normal_quantiles > cut
        centre_mask = np.full(below_min_mask.shape, True)
        centre_mask[below_min_mask] = False
        centre_mask[above_max_mask] = False

        central_theoretical_quantiles = normal_quantiles[centre_mask]
        central_observed_quantiles = observed_quantile_estimates[centre_mask]

        map_unc, map_off = np.polyfit(x=central_theoretical_quantiles, y=central_observed_quantiles, deg=1)

        return map_unc

    @staticmethod
    def calculate_sigma_s_m_batched(_means, _arrays, _sigma_is_array):
        sigma_ms = np.zeros(_means.shape)

        for x in np.ndindex(*_means.shape):
            _mean = np.array((_means[x],))
            _array = _arrays[:, x, ].flatten()
            _sigma_i = _sigma_is_array.flatten()

            result_root = optimize.root(
                partial(Model.differentiated_log_liklihood, est_mu=_mean, obs_vals=_array, obs_error=_sigma_i),
                x0=np.power(2.0, -20),
            )

            sigma_ms[x] = result_root

        return np.abs(sigma_ms)

    @staticmethod
    def calculate_sigma_s_m(mean: np.array, arrays: np.array, sigma_is_array: np.array, process_local):
        # Maximise liklihood of data at m under normal(mu_m, sigma_i + sigma_s_m) by optimising sigma_s_m
        # mean[m]
        # arrays[n,m]
        # sigma_i_array[n]
        #

        # func = partial(Model.differentiated_log_liklihood, est_mu=mean, obs_vals=arrays, obs_error=sigma_is_array)
        #
        # shape = mean.shape
        # num = len(sigma_is_array)

        # sigma_ms = Model.vectorised_optimisation_bisect(func,
        #                                                 0,
        #                                                 20,
        #                                                 31,
        #                                                 arrays.shape
        #                                                 )

        # print(sigma_ms.shape)
        # print([np.max(sigma_ms), np.min(sigma_ms), np.std(sigma_ms), np.mean(sigma_ms)])
        #
        # print(mean.shape)
        # print(arrays.shape)
        # print(sigma_ms.shape)
        #
        means_batches = np.array_split(mean, 12)
        arrays_batches = np.array_split(arrays, 12, axis=1)

        sigma_ms_batches = process_local(
            [
                partial(
                    Model.calculate_sigma_s_m_batched,
                    mean_batch,
                    array_batch,
                    sigma_is_array,
                )
                for mean_batch, array_batch
                in zip(means_batches, arrays_batches)
            ]
        )

        sigma_ms = np.concatenate(sigma_ms_batches).flatten()

        # sigma_ms = np.zeros(mean.shape)
        # for x in np.ndindex(*mean.shape):
        #     # print("#######")
        #     # print([x])
        #     # print(f"Vectorised bisec gives: {sigma_ms[x]}")
        #     _mean = np.array((mean[x],))
        #     # print(_mean)
        #     _array = arrays[:, x, ].flatten()
        #     # print(_array)
        #     _sigma_i = sigma_is_array.flatten()
        #     # print(_sigma_i)
        #     #
        #     # print([_sigma_i.shape, _mean.shape, _array.shape, Model.log_liklihood(np.array((1.0,)), _mean, _array, _sigma_i)])
        #
        #     # result = shgo(partial(Model.log_liklihood, est_mu=_mean, obs_vals=_array, obs_error=_sigma_i))
        #     # start = time.time()
        #     result_root = optimize.root(
        #         partial(Model.differentiated_log_liklihood, est_mu=_mean, obs_vals=_array, obs_error=_sigma_i),
        #         x0=np.power(2.0, -20),
        #     )
        #     # finish = time.time()
        #     # print(f"Root found in {finish-start}")
        #
        #     # start = time.time()
        #     # result_min = optimize.minimize(
        #     #     partial(Model.log_liklihood, est_mu=_mean, obs_vals=_array, obs_error=_sigma_i),
        #     #     np.array((np.power(2.0, -20.0),),),
        #     #     bounds=((0.0, 20.0,),)
        #     # )
        #     # finish = time.time()
        #     # print(f"Minimised in {finish-start}")
        #
        #     # print([sigma_ms_bisect[x], result_root.x, result_min.x,])
        #     # print([result.x, sigma_ms[x], result.fun])
        #     sigma_ms[x] = np.abs(result_root.x)

        # sigma_ms = Model.maximise_over_range(func,
        #                                      0,
        #                                      6,
        #                                      150,
        #                                      arrays.shape)

        return sigma_ms

    @staticmethod
    def calculate_sigma_s_m_np(mean: np.array, arrays: np.array, sigma_is_array: np.array, process_local):
        # Maximise liklihood of data at m under normal(mu_m, sigma_i + sigma_s_m) by optimising sigma_s_m
        # mean[m]
        # arrays[n,m]
        # sigma_i_array[n]
        #

        func = partial(Model.differentiated_log_liklihood, est_mu=mean, obs_vals=arrays, obs_error=sigma_is_array)

        sigma_ms = Model.vectorised_optimisation_bisect(func,
                                                        0,
                                                        20,
                                                        31,
                                                        arrays.shape
                                                        )

        return sigma_ms

    @staticmethod
    def maximise_over_range(func, start, stop, num, shape):
        xs = np.linspace(start, stop, num)

        x_opt = np.ones(shape[1], dtype=np.float32) * xs[0]  # n
        x_current = np.ones(shape[1], dtype=np.float32) * xs[0]  # n

        y_max = func(x_current[np.newaxis, :])  # n -> 1,n -> n

        for x in xs[1:]:
            x_current[:] = x  # n
            y = func(x_current[np.newaxis, :])  # n -> 1,n -> n
            y_above_y_max_mask = y > y_max  # n
            # y_max[y_above_y_max_mask] = y[y_above_y_max_mask]
            x_opt[y_above_y_max_mask] = x

        return x_opt

    @staticmethod
    def vectorised_optimisation_bf(func, start, stop, num, shape):
        xs = np.linspace(start, stop, num)

        val = np.ones(shape, dtype=np.float32) * xs[0] + 1.0 / 10000000000000000000000.0
        res = np.ones((shape[1], shape[2], shape[3]), dtype=np.float32) * xs[0]

        y_max = func(val)

        for x in xs[1:]:
            val[:, :, :, :] = 1
            val = val * x

            y = func(x)

            y_above_y_max_mask = y > y_max
            y_max[y_above_y_max_mask] = y[y_above_y_max_mask]
            res[y_above_y_max_mask] = x

        return res

    @staticmethod
    def vectorised_optimisation_bisect(func, start, stop, num, shape):
        # Define step 0
        x_lower_orig = (np.ones(shape[1], dtype=np.float32) * start)

        x_upper_orig = np.ones(shape[1], dtype=np.float32) * stop

        f_lower = func(x_lower_orig[np.newaxis, :])

        f_upper = func(x_upper_orig[np.newaxis, :])

        test_mat = f_lower * f_upper

        test_mat_mask = test_mat > 0

        # mask = np.tile(test_mat_mask, (x_lower_orig.shape[0], 1))

        x_lower = x_lower_orig
        x_upper = x_upper_orig

        for i in range(num):
            x_bisect = x_lower + ((x_upper - x_lower) / 2)

            f_lower = func(x_lower[np.newaxis, :])
            f_bisect = func(x_bisect[np.newaxis, :])

            f_lower_positive = f_lower >= 0
            f_lower_negative = f_lower < 0

            f_bisect_positive = f_bisect >= 0
            f_bisect_negative = f_bisect < 0

            f_lower_positive_bisect_negative = f_lower_positive * f_bisect_negative
            f_lower_negative_bisect_positive = f_lower_negative * f_bisect_positive

            f_lower_positive_bisect_positive = f_lower_positive * f_bisect_positive
            f_lower_negative_bisect_negative = f_lower_negative * f_bisect_negative

            x_upper[f_lower_positive_bisect_negative] = x_bisect[f_lower_positive_bisect_negative]
            x_upper[f_lower_negative_bisect_positive] = x_bisect[f_lower_negative_bisect_positive]

            x_lower[f_lower_positive_bisect_positive] = x_bisect[f_lower_positive_bisect_positive]
            x_lower[f_lower_negative_bisect_negative] = x_bisect[f_lower_negative_bisect_negative]

        # Replace original vals
        x_lower[test_mat_mask] = 0.0

        return x_lower

    @staticmethod
    def differentiated_log_liklihood(est_sigma, est_mu=0.0, obs_vals=0.0, obs_error=0.0):
        """Calculate the value of the differentiated log likelihood for the values of mu, sigma"""

        term1 = np.square(obs_vals - est_mu) / np.square(np.square(est_sigma) + np.square(obs_error))
        term2 = np.ones(est_sigma.shape, dtype=np.float32) / (np.square(est_sigma) + np.square(obs_error))

        difference = np.sum(term1, axis=0) - np.sum(term2, axis=0)

        return difference

    @staticmethod
    def log_liklihood(est_sigma, est_mu=0.0, obs_vals=0.0, obs_error=0.0):
        term_1 = -0.5 * ((np.square(obs_vals) - np.square(est_mu)) / (np.square(est_sigma) + np.square(obs_error)))
        term_2 = -0.5 * (np.log(2 * np.pi * (np.square(est_sigma) + np.square(obs_error))))

        return np.sum(term_1, axis=0) + np.sum(term_2, axis=0)

    def evaluate(self, xmap: Xmap, dtag: Dtag):
        xmap_array = np.copy(xmap.to_array())

        if xmap_array.shape != self.mean.shape:
            raise Exception("Wrong shape!")

        residuals = (xmap_array - self.mean)
        denominator = (np.sqrt(np.square(self.sigma_s_m) + np.square(self.sigma_is[dtag])))

        return residuals / denominator

    @staticmethod
    def liklihood(est_sigma, est_mu, obs_vals, obs_error):
        term1 = -np.square(obs_vals - est_mu) / (2 * (np.square(est_sigma) + np.square(obs_error)))
        term2 = np.log(np.ones(est_sigma.shape, dtype=np.float32) / np.sqrt(
            2 * np.pi * (np.square(est_sigma) + np.square(obs_error))))
        return np.sum(term1 + term2)

    @staticmethod
    def log_liklihood_normal(est_sigma, est_mu, obs_vals, obs_error):
        n = obs_error.size

        # term1 = -np.square(obs_vals - est_mu) / (2 * (np.square(est_sigma) + np.square(obs_error)))
        # term2 = np.log(np.ones(est_sigma.shape) / np.sqrt(2 * np.pi * (np.square(est_sigma) + np.square(obs_error))))

        term1 = -1 * (n / 2) * np.log(2 * np.pi)
        term2 = -1 * (1 / 2) * np.log(np.square(est_sigma) + np.square(obs_error))  # n, m
        term3 = -1 * (1 / 2) * (1 / (np.square(est_sigma) + np.square(obs_error))) * np.square(obs_vals - est_mu)  # n,m

        return term1 + np.sum(term2 + term3, axis=0)  # 1 + m

    def save_maps(self, pandda_dir: Path, shell: Shell, grid: Grid, p1: bool = True):
        # Mean map
        mean_array = self.mean

        mean_grid = gemmi.FloatGrid(*mean_array.shape)
        mean_grid_array = np.array(mean_grid, copy=False)
        mean_array_typed = mean_array.astype(mean_grid_array.dtype)
        mean_grid_array[:, :, :] = mean_array_typed[:, :, :]
        mean_grid.spacegroup = grid.grid.spacegroup
        mean_grid.set_unit_cell(grid.grid.unit_cell)

        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = mean_grid
        if p1:
            ccp4.grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
        else:
            ccp4.grid.symmetrize_max()
        ccp4.update_ccp4_header(2, True)
        ccp4.write_ccp4_map(str(pandda_dir / PANDDA_MEAN_MAP_FILE.format(number=shell.number,
                                                                         res=shell.res_min.resolution,
                                                                         )))

        # sigma_s_m map
        sigma_s_m_array = self.sigma_s_m
        sigma_s_m_grid = gemmi.FloatGrid(*sigma_s_m_array.shape)
        sigma_s_m_grid_array = np.array(sigma_s_m_grid, copy=False)
        sigma_s_m_array_typed = sigma_s_m_array.astype(sigma_s_m_grid_array.dtype)
        sigma_s_m_grid_array[:, :, :] = sigma_s_m_array_typed[:, :, :]
        sigma_s_m_grid.spacegroup = grid.grid.spacegroup
        sigma_s_m_grid.set_unit_cell(grid.grid.unit_cell)

        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = sigma_s_m_grid
        if p1:
            ccp4.grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
        else:
            ccp4.grid.symmetrize_max()
        ccp4.update_ccp4_header(2, True)
        ccp4.write_ccp4_map(str(pandda_dir / PANDDA_SIGMA_S_M_FILE.format(
            number=shell.number,
                                                                          res=shell.res_min.resolution,
                                                                          )))


@dataclasses.dataclass()
class Zmap:
    zmap: gemmi.FloatGrid

    @staticmethod
    def from_xmap(model: Model, xmap: Xmap, dtag: Dtag, model_number=0, debug=False):

        # Get zmap
        zmap_array = model.evaluate(xmap, dtag)

        # Symmeterize Positive values
        new_zmap_grid_pos = xmap.new_grid()
        new_zmap_array_pos = np.array(new_zmap_grid_pos, copy=False)
        new_zmap_array_pos[:, :, :] = zmap_array[:, :, :]
        new_zmap_grid_pos.symmetrize_max()

        # Symmetrize Negative values
        new_zmap_grid_neg = xmap.new_grid()
        new_zmap_array_neg = np.array(new_zmap_grid_neg, copy=False)
        new_zmap_array_neg[:, :, :] = -zmap_array[:, :, :]
        new_zmap_grid_neg.symmetrize_max()

        # Get merged map
        new_zmap_grid_symm = xmap.new_grid()
        new_zmap_array_symm = np.array(new_zmap_grid_symm, copy=False)
        new_zmap_array_symm[:, :, :] = new_zmap_array_pos[:, :, :] - new_zmap_array_neg[:, :, :]

        # Get mean and std
        zmap_mean = np.mean(zmap_array)
        zmap_std = np.std(zmap_array)

        zmap_symm_mean = np.mean(new_zmap_array_symm)
        zmap_symm_std = np.std(new_zmap_array_symm)

        zmap_sparse_mean = np.mean(zmap_array[zmap_array != 0.0])
        zmap_sparse_std = np.std(zmap_array[zmap_array != 0.0])
        if debug:
            print(f"\t\tZmap mean is: {zmap_sparse_mean}")
            print(f"\t\tZmap mean is: {zmap_sparse_std}")
            print(f"\t\tZmap max is: {np.max(zmap_array[zmap_array != 0.0])}")
            print(f"\t\tZmap >1 is: {zmap_array[zmap_array > 1.0].size}")
            print(f"\t\tZmap >2 is: {zmap_array[zmap_array > 1.0].size}")
            print(f"\t\tZmap >2.5 is: {zmap_array[zmap_array > 2.5].size}")
            print(f"\t\tZmap >3 is: {zmap_array[zmap_array > 3.0].size}")

            f = sns.displot(x=zmap_array[zmap_array != 0.0], kind="ecdf")
            f.set(xlim=(-10,10))
            f.savefig(f"{model_number}.png")

        normalised_zmap_array = (zmap_array - zmap_sparse_mean) / zmap_sparse_std

        zmap = Zmap.grid_from_template(xmap, normalised_zmap_array)
        return Zmap(zmap)

    @staticmethod
    def grid_from_template(xmap: Xmap, zmap_array: np.array):
        spacing = [xmap.xmap.nu, xmap.xmap.nv, xmap.xmap.nw, ]
        new_grid = gemmi.FloatGrid(*spacing)
        # new_grid.spacegroup = xmap.xmap.spacegroup
        new_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
        new_grid.set_unit_cell(xmap.xmap.unit_cell)

        new_grid_array = np.array(new_grid, copy=False)
        new_grid_array[:, :, :] = zmap_array[:, :, :]

        return new_grid

    @staticmethod
    def grid_from_grid_template(xmap: gemmi.FloatGrid, zmap_array: np.array):
        spacing = [xmap.nu, xmap.nv, xmap.nw, ]
        new_grid = gemmi.FloatGrid(*spacing)
        # new_grid.spacegroup = xmap.xmap.spacegroup
        new_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
        new_grid.set_unit_cell(xmap.unit_cell)

        new_grid_array = np.array(new_grid, copy=False)
        new_grid_array[:, :, :] = zmap_array[:, :, :]

        return new_grid

    def to_array(self, copy=True):
        return np.array(self.zmap, copy=copy)

    def shape(self):
        return [self.zmap.nu, self.zmap.nv, self.zmap.nw]

    def spacegroup(self):
        return self.zmap.spacegroup

    def unit_cell(self):
        return self.zmap.unit_cell

    def save(self, path: Path, p1: bool = True):
        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = self.zmap
        if p1:
            ccp4.grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
        else:
            ccp4.grid.symmetrize_max()
        ccp4.update_ccp4_header(2, True)
        ccp4.write_ccp4_map(str(path))

    def __getstate__(self):
        return XmapPython.from_gemmi(self.zmap)

    def __setstate__(self, zmap_pyhton: XmapPython):
        self.zmap = zmap_pyhton.to_gemmi()


@dataclasses.dataclass()
class Zmaps:
    zmaps: typing.Dict[Dtag, Zmap]

    @staticmethod
    def from_xmaps(model: Model, xmaps: Xmaps, model_number=0, debug=False):
        zmaps = {}
        for dtag in xmaps:
            xmap = xmaps[dtag]
            zmap = Zmap.from_xmap(model, xmap, dtag, model_number=model_number, debug=debug)
            zmaps[dtag] = zmap

        return Zmaps(zmaps)

    def __len__(self):
        return len(self.zmaps)

    def __iter__(self):
        for dtag in self.zmaps:
            yield dtag

    def __getitem__(self, item):
        return self.zmaps[item]
