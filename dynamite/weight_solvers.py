import os
import copy
import time
import numpy as np
from astropy import table
from astropy.io import ascii
import subprocess
import logging
from typing import NamedTuple
from scipy import optimize

try:
    import cvxopt
    from scipy.linalg.lapack import dpotrf, dpotrs
except ModuleNotFoundError:
    pass

from scipy.linalg.blas import dsyrk, dtrmv

try:
    # adelie's rayon thread pool mis-detects core count on high-core-count
    # machines and narrows this process's own CPU affinity to a single core
    # as a side effect of import; every process forked afterwards (pool
    # workers, orblib chunks) would otherwise inherit that one-core mask.
    # Restore what we had rather than opening up to os.cpu_count(): that is
    # the whole machine, so it would also discard a restriction the user
    # meant to impose (taskset is not cgroup-enforced, so nothing else
    # would put it back).
    _saved_affinity = (
        os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
    )
    import adelie.solver as _adelie_solver
    import adelie.matrix as _adelie_matrix

    _ADELIE_AVAILABLE = True
    if _saved_affinity is not None:
        os.sched_setaffinity(0, _saved_affinity)
except ImportError:
    _ADELIE_AVAILABLE = False

from dynamite import constants
from dynamite import analysis
from dynamite import kinematics as dyn_kin


class AdelieProblem(NamedTuple):
    """Everything the adelie path needs, built without materializing A.

    ``X`` has exactly n_rows rows: the sqrt(mu) penalty row REPLACES A's
    total-mass row, which survives only as the small vector ``row0_vec``
    (bitwise equal to A[0], i.e. ones/econ[0])."""

    X: np.ndarray  # (n_rows, n_orbs) F-order, column-scaled
    col_norm: np.ndarray  # (n_orbs,) unit-L2 norms incl. penalty row
    y: np.ndarray  # (n_rows,) ALM target; slot 0 updated per iter
    row0_vec: np.ndarray  # (n_orbs,) == A[0] bitwise
    b0: float  # rhs[0] = total_mass/total_mass_error


class GramProblem(NamedTuple):
    """Everything the blockwise Gram path needs to run cvxopt/ADMM, without
    ever materializing ``A``.

    ``P``/``q``/``col_norm``/``b_max`` are exactly what the classic
    cvxopt/admm branches compute from a materialized ``A_rest`` (see
    :meth:`NNLS.solve`); here they come from
    :meth:`NNLS.construct_gram_and_rhs_blockwise` instead, one row block at
    a time. ``G``/``v``/``b_sq_rest`` are the RAW (pre column-scaling)
    accumulator outputs, kept alongside ``P``/``q`` so chi2 can be read off
    the quadratic form ``w'Gw - 2 w'v + ||b||^2`` after solving, instead of
    a second pass over ``A``. ``A_mass``/``b_mass`` is the small
    ``(n_intrinsic + n_apertures, n_orbs)`` mass-constraint block, which
    IS kept materialized (it is orders of magnitude smaller than the
    kinematic rows) purely so ``chi2_kin = chi2_rest - chi2_mass`` can be
    split out without a second ``n_orbs x n_orbs`` Gram matrix."""

    P: np.ndarray  # (n_orbs, n_orbs) unit-L2-column-scaled Gram matrix
    q: np.ndarray  # (n_orbs,) unit-L2-column-scaled cross term
    col_norm: np.ndarray  # (n_orbs,) == np.linalg.norm(A_rest, axis=0)
    b_max: float
    G: np.ndarray  # (n_orbs, n_orbs) raw Gram matrix, A_rest^T A_rest
    v: np.ndarray  # (n_orbs,) raw cross term, A_rest^T b_rest
    b_sq_rest: float  # ||b_rest||^2, rest = all rows except row 0
    A_mass: np.ndarray  # (n_intrinsic+n_apertures, n_orbs), econ-divided
    b_mass: np.ndarray  # (n_intrinsic+n_apertures,)
    row0_vec: np.ndarray  # (n_orbs,) == A[0], i.e. ones/econ[0]
    b0: float  # rhs[0] = total_mass/total_mass_error


def _apply_diagonal_ridge(P, lam):
    """``P += lam * mean(diag(P))`` IN PLACE - Vasiliev Eq. 7's diagonal
    regularisation, dimensionless in lam so one value transfers across p.

    Returns the ABSOLUTE shift added per diagonal entry; callers that later
    recover quadratic forms from an ADMM Cholesky factor must pass it as
    ``gram_quadratic_form``'s ``extra_shift``, or the bookkeeping
    double-counts the ridge (pinned in dev_tests/test_admm_free_p.py). A
    no-op at lam <= 0 (returns 0.0, P untouched)."""
    if lam <= 0.0:
        return 0.0
    p = P.shape[0]
    scale = float(np.trace(P)) / p
    P.flat[:: p + 1] += lam * scale
    return lam * scale


class NormalEquationAccumulator:
    """Blockwise accumulation of the cvxopt/ADMM normal equations, so the
    design matrix ``A`` never has to exist all at once.

    Feed it row blocks of ``A[1:]`` (each already divided by ITS OWN econ
    slice — see the ordering note in
    :meth:`NNLS.construct_gram_and_rhs_blockwise`) and the matching slice of
    ``b[1:]``. It accumulates the RAW Gram matrix and cross term

        G = A[1:]^T A[1:]        v = A[1:]^T b[1:]

    via one ``dsyrk`` rank-k update per block (no ``(n_orbs, n_orbs)``
    temporary — the update is done in place into the running ``G``), then
    on :meth:`finalize` the column norms fall out of the diagonal — no
    separate pass, and nothing is subtracted anywhere, which is exactly why
    this survives a column-norm spread of many decades without cancellation:

        col = sqrt(diag(G));  P = G / outer(col, col);  q = -v / (col * b_max)

    Adapted from the validated, self-tested design in
    ``PM_grid/_diag_09_blockwise/blockwise_normal_eq.py`` and
    ``WIRING.md`` (self-test there: col_norm median rel 1.94e-15 / max
    8.83e-15 vs a materialized computation; P max rel 1.78e-14; q max rel
    1.63e-14). This is a standalone copy — dynamite does not import
    anything from PM_grid — with one addition: ``b_sq_sum``, needed here to
    recover chi2 from the Gram form without a second pass over ``A``.
    """

    def __init__(self, n_orbs, dtype=np.float64):
        self.n_orbs = n_orbs
        self.dtype = dtype
        # symmetric: only the upper triangle is filled until finalize()
        self.G = np.zeros((n_orbs, n_orbs), dtype=dtype, order="F")
        self.v = np.zeros(n_orbs, dtype=np.float64)
        self.b_sq_sum = 0.0
        self.b_max = 0.0
        self.n_rows = 0

    def add(self, A_block, b_block):
        """``A_block``: ``(rows, n_orbs)``, already divided by its own econ
        rows. ``b_block``: ``(rows,)``, likewise."""
        A_block = np.asarray(A_block, dtype=self.dtype)
        assert A_block.shape[1] == self.n_orbs, A_block.shape
        # dsyrk does the rank-k update in place: no (n_orbs, n_orbs) temporary
        self.G = dsyrk(
            1.0, A_block, beta=1.0, c=self.G, trans=1, lower=0, overwrite_c=1
        )
        b_block = np.asarray(b_block, dtype=np.float64)
        self.v += A_block.T @ b_block
        self.b_sq_sum += float(np.dot(b_block, b_block))
        self.b_max = max(self.b_max, float(np.abs(b_block).max(initial=0.0)))
        self.n_rows += A_block.shape[0]

    def finalize(self):
        """Return ``(P, q, col_norm, b_max)`` with unit-L2 column scaling.

        Mirrors ``self.G`` to a full symmetric matrix in place (only the
        upper triangle is filled by ``dsyrk``); ``self.G``/``self.v``/
        ``self.b_sq_sum`` remain available afterwards for a raw-scale chi2.
        """
        iu = np.triu_indices(self.n_orbs)
        self.G[(iu[1], iu[0])] = self.G[iu]  # mirror to full symmetric
        col = np.sqrt(np.abs(np.diag(self.G)).copy())
        col[col == 0] = 1.0  # null orbits: leave alone
        b_max = self.b_max if self.b_max > 0 else 1.0
        P = self.G / np.outer(col, col)
        q = -self.v / (col * b_max)
        return P, q, col, b_max


def _downcast_orblib(orblib, dtype):
    """Downcast the retained orbit-library data to ``dtype``, in place.

    This is most of the resident memory of a solve, not the matrix itself.
    No-op unless ``dtype`` is float32. ``None`` entries are tolerated so the
    streamed path can call it per set as each one arrives; ``copy=False``
    makes re-converting an already-downcast set free.
    """
    if dtype != np.float32:
        return
    for hist in orblib.vel_histograms:
        if hist is not None:
            hist.y = hist.y.astype(np.float32, copy=False)
    intrinsic = getattr(orblib, "intrinsic_masses", None)
    if intrinsic is not None:
        orblib.intrinsic_masses = intrinsic.astype(np.float32, copy=False)
    projected = getattr(orblib, "projected_masses", None)
    if projected is not None:
        orblib.projected_masses = [
            p.astype(np.float32, copy=False) if p is not None else None
            for p in projected
        ]


def _scale_columns(X, b_rest, dtype):
    """Unit-L2 scale ``X``'s columns in place; returns ``(col_norm, y)``.

    Shared by both augmented-matrix builders (`_build_augmented_X` from an
    existing A, `construct_adelie_matrix_and_rhs` from the orbit library) so
    they cannot drift. Zero columns are left unscaled.
    """
    col_norm = np.linalg.norm(X, axis=0)
    col_norm[col_norm == 0] = 1.0
    X /= col_norm
    y = np.concatenate([[0.0], b_rest]).astype(dtype)
    return col_norm, y


def chi2_vector_from_residuals(resid_full, row0_sq):
    """Per-row squared residuals of ``A @ w - b`` without materializing A.

    ``resid_full`` holds rows 1..n of that residual (from adelie's
    ``state.resid``, which is exactly the plain residual on rows 1.. because
    ``y[1:] == b_rest``); the total-mass row's contribution arrives separately
    as ``row0_sq = (A[0] @ w - b[0])**2`` because X replaces that row with the
    ALM penalty row. Index 0 of the result is ``row0_sq`` itself, keeping the
    indexing of the A-based chi2_vector (chi2_kin slices
    ``[n_mass_constraints:]``).

    Algebraically identical to ``(A @ w - b)**2``; differs only in rounding,
    since the gemv over A is replaced by the solver's accumulated residual.
    """
    resid_full = np.asarray(resid_full, dtype=np.float64).ravel()
    return np.concatenate(([row0_sq], resid_full[1:] ** 2))


class WeightSolver(object):
    """Generic WeightSolver class

    Specific implementations are defined as sub-classes. Each one should
    have a main method `solve`

    Parameters
    ----------
    config : a ``dyn.config_reader.Configuration`` object
    model : a ``dyn.model.Model`` object
    CRcut : Bool, default False
        whether to use the `CRcut` solution for the counter-rotating orbit
        problem. See Zhu et al. 2018 for more. If `CRcut` is given in the
        configuration file's weight solver settings (which is normally the
        case), this parameter is ignored.

    """

    def __init__(self, config, model, CRcut=False):
        self.logger = logging.getLogger(f"{__name__}.{__class__.__name__}")
        self.config = config
        self.system = config.system
        self.settings = config.settings.weight_solver_settings
        self.model = model
        self.direc_with_ml = model.directory
        self.direc_no_ml = model.directory_noml
        if "CRcut" in self.settings.keys():
            CRcut = self.settings["CRcut"]
        self.CRcut = CRcut
        self.weight_file = f"{self.direc_with_ml}{constants.weight_file}"

    def solve(self, orblib, ignore_existing_weights=False):
        """Template solve method

        Specific implementations should override this.

        Parameters
        ----------
        orblib : dyn.OrbitLibrary object
        ignore_existing_weights : bool
            If True, do not check for already existing weights and solve again.
            Default is False.

        Returns
        -------
        weights : array
            orbit weights
        chi2_all : float
            a total chi2 value
        chi2_kin : float
            a chi2 value purely for kinematics
        chi2_kinmap : float
            directly calculates the chi2 from the kinematic maps
        """
        self.logger.info(f"Using WeightSolver: {__class__.__name__}")
        # ...
        # calculate orbit weights, and model chi2 values here
        # ...
        weights = 0.0
        chi2_tot = 0.0
        chi2_kin = 0.0
        chi2_kinmap = 0.0
        # ...
        return weights, chi2_tot, chi2_kin, chi2_kinmap

    def chi2_kinmap(self, weights, orblib=None):
        """
        Returns the chi2 directly calculated from the gh kinematic maps.

        For each kinematic set, the following applies: If number_GH in the
        weight_solver_settings is smaller than the number of GH coefficients
        in the data file, only number_GH coefficients will be considered.
        If number_GH is greater than the number of GH coefficients in the
        data file, only the coefficients in the data file will be considered.

        Does only work with Gauss Hermite kinematics.

        Parameters
        ----------
        weights : ``numpy.array`` like
            The model's orbital weights.
        orblib : ``dyn.orblib.OrbitLibrary``, optional
            An orbit library whose velocity histograms are UNMUTATED (see the
            note in the code below). Pass one to avoid re-reading the library
            from disk, which dominates the cost of this method. Default is
            None, i.e. read a fresh library.

        Returns
        -------
        chi2_kinmap : float
            chi2 directly calculated from the kinematic maps: sum of
            squared residuals of V, sigma, and GH coefficients from h_3 to h_N

        """
        stars = self.system.get_unique_triaxial_visible_component()
        if any(k.type != "GaussHermite" for k in stars.kinematic_data):
            self.logger.info(
                "All kinematics must be 'GaussHermite' for kinmapchi2. Value set to nan."
            )
            return float("nan")  # #######################################
        number_gh = self.settings["number_GH"]
        chi2_kinmap = 0.0
        # NOTE: the orbit library used here must have UNMUTATED velocity
        # histograms. construct_nnls_matrix_and_rhs zeroes the first and last
        # velocity bin of each 1D histogram in place, to mimic
        # `triaxnnls_CRcut.f90`; reusing a library still in that state changes
        # chi2_kinmap's value (measured: ~7% off, with weights, chi2_tot and
        # chi2_kin all identical, which is exactly the kind of discrepancy
        # nobody would trace back to here).
        # construct_nnls_matrix_and_rhs restores those two edge slices before
        # returning, so a caller that has already built the NNLS matrix can
        # hand us that same library instead of paying for a second full
        # read - the read is the single most expensive step in a model.
        if orblib is None:
            orblib = self.model.get_orblib()
            orblib.read_vel_histograms()
        for kin_set, kin_data in enumerate(stars.kinematic_data):
            n_gh = min(number_gh, kin_data.max_gh_order)
            coefs = ["v", "sigma"] + [f"h{i}" for i in range(3, n_gh + 1)]
            # get the model's projected masses=flux (unused) and kinematic data
            a = analysis.Analysis(config=self.config, model=self.model, kin_set=kin_set)
            model_gh_coef = a.get_gh_model_kinematic_maps(
                v_sigma_option="fit", weights=weights, orblib=orblib
            )
            # get the observed projected masses (unused) and kinematic data
            kinematics_data = kin_data.get_data()
            # calculate chi2_kinmap
            for coef in coefs:
                obs_val = np.array(kinematics_data[coef])
                mod_val = np.array(model_gh_coef[coef])
                err_val = np.array(kinematics_data["d" + coef])
                chi2_kinmap += sum(np.square((obs_val - mod_val) / err_val))
        return chi2_kinmap

    def weight_file_exists(self):
        """Check whether the file(s) holding the current model's weights exist.

        May be re-implemented by sub-classes.

        Returns
        -------
        bool
            True if weight solving data exists, False otherwise.

        """
        return os.path.isfile(self.weight_file)


class LegacyWeightSolver(WeightSolver):
    """Use `legacy` AKA Fortran weight solving.

    Uses the legcay_fortran program ``triaxnnls_CRcut.f90`` or
    ```triaxnnls_noCRcut.f90``. Uses Lawson and Hanson non-negative
    least-squares algorithm.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(f"{__name__}.{__class__.__name__}")
        self.legacy_directory = self.config.settings.legacy_settings["directory"]
        self.sformat = self.system.parameters[0].sformat  # this is ml's format
        ml_idx = self.direc_with_ml.rindex("/ml")
        ml_str = self.direc_with_ml[ml_idx + 3 :]
        self.ml = float(ml_str[:-1]) if ml_str[-1] == "/" else float(ml_str)
        self.fname_nn_kinem = self.direc_with_ml + "nn_kinem.out"
        self.fname_nn_nnls = self.direc_with_ml + "nn_nnls.out"
        # check the format of the orbit library files
        # check == True means there are 2 orblib_* and 2 orblibbox_* files
        # check == False means there are only two orblib files,
        # orblib.dat.bz2 and orbibbox.dat.bz2 (legacy behavior)
        pth = self.direc_no_ml + "datfil/"
        check = (
            os.path.isfile(f"{pth}orblib_qgrid.dat.bz2")
            and os.path.isfile(f"{pth}orblib_losvd_hist.dat.bz2")
            and os.path.isfile(f"{pth}orblibbox_qgrid.dat.bz2")
            and os.path.isfile(f"{pth}orblibbox_losvd_hist.dat.bz2")
        )
        self.legacy_files = False if check else True
        # prepare fortran input file for nnls
        self.copy_kinematic_data()
        self.create_fortran_input_nnls()
        self.logger.info(
            f"{__class__.__name__} is DEPRECATED and "
            "will be removed in a future version of "
            f"DYNAMITE. Use weight solver type NNLS instead "
            f"of {__class__.__name__} if you can."
        )

    def copy_kinematic_data(self):
        """Copy kin data to infil/ direc"""
        stars = self.system.get_unique_triaxial_visible_component()
        kinematics = stars.kinematic_data
        # convert kinematics to old format to input to fortran
        for i in np.arange(len(kinematics)):
            if len(kinematics) == 1:
                old_filename = self.direc_no_ml + "infil/kin_data.dat"
            else:
                old_filename = self.direc_no_ml + "infil/kin_data_" + str(i) + ".dat"
            kinematics[i].convert_to_old_format(old_filename, self.settings)
        # combine all kinematics into one file
        if len(kinematics) > 1:
            if not all(isinstance(kin, dyn_kin.GaussHermite) for kin in kinematics):
                text = "Multiple kinematics: all must be GaussHermite"
                self.logger.error(text)
                raise ValueError(text)
            # make a dummy 'kins_combined' object ...
            kins_combined = copy.deepcopy(kinematics[0])
            # ...replace data attribute with stacked table of all kinematics
            kins_combined.data = table.vstack([k.get_data() for k in kinematics])
            kins_combined.n_apertures = len(kins_combined.data)
            kins_combined.max_gh_order = self.settings["number_GH"]
            old_filename = self.direc_no_ml + "infil/kin_data_combined.dat"
            kins_combined.convert_to_old_format(old_filename, self.settings)

    def create_fortran_input_nnls(self):
        """create fortran input file nn.in

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # When varying ml the LOSVD is scaled - no new orbits are calculated.
        # Therefore we need to know the ml that was used for the orbit library.
        # The scaling factor is sqrt(model_ml/original_orblib_ml).
        ml_scaling_factor = self.config.all_models.get_model_velocity_scaling_factor(
            model=self.model
        )
        # -------------------
        # write nn.in
        # -------------------
        n_kin = len(self.system.get_unique_triaxial_visible_component().kinematic_data)

        if n_kin == 1:
            kin_data_file = "kin_data.dat"

        else:
            kin_data_file = "kin_data_combined.dat"

        text = (
            "infil/parameters_pot.in"
            + "\n"
            + str(self.settings["regularisation"])
            + "                                  [ regularization strength, 0 = no regularization ]"
            + "\n"
            + f"ml{self.ml:{self.sformat}}/nn\n"
            + "datfil/mass_qgrid.dat"
            + "\n"
            + "datfil/mass_aper.dat"
            + "\n"
            + str(self.settings["number_GH"])
            + "	                           [ # of GH moments to constrain the model]"
            + "\n"
            + "infil/"
            + kin_data_file
            + "\n"
            + str(self.settings["lum_intr_rel_err"])
            + "                               [ relative error for intrinsic luminosity ]"
            + "\n"
            + str(self.settings["sb_proj_rel_err"])
            + "                               [ relative error for projected SB ]"
            + "\n"
            + str(ml_scaling_factor)
            + "                                [ scale factor related to M/L, sqrt( (M/L)_k / (M/L)_ref ) ]"
            + "\n"
        )
        if self.legacy_files:
            text += (
                2 * f"datfil/orblib_{self.ml}.dat\n"
                + 2 * f"datfil/orblibbox_{self.ml}.dat\n"
            )  # yes, really...
        else:
            for f in "_qgrid", "_losvd_hist", "box_qgrid", "box_losvd_hist":
                text += f"datfil/orblib{f}_{self.ml}.dat\n"
        text += (
            str(self.settings["nnls_solver"])
            + "                                  [ nnls solver ]"
        )

        nn_file = open(self.direc_no_ml + f"ml{self.ml:{self.sformat}}/nn.in", "w")
        nn_file.write(text)
        nn_file.close()

    def solve(self, orblib=None, ignore_existing_weights=False):
        """Main method to solve NNLS problem.

        Parameters
        ----------
        orblib : dyn.OrbitLibrary
            This parameter is not used in this Legacy implementation (as all
            orbit library information is read from files). It is included here
            for consistency with later WeightSolver implementations
        ignore_existing_weights : bool
            If True, do not check for already existing weights and solve again.
            Default is False.

        Returns
        -------
        tuple
            (weights, chi2_all, chi2_kin, chi2_kinmap) where:
                -   weights : array, of orbit weights
                -   chi2_all : float, sum of squared residuals for intrinsic
                    masses, projected_masses and GH coefficients from h_1 to h_n
                -   chi2_kin : float sum of squared residuals for GH
                    coefficients h_1 to h_n
                -   chi2_kinmap : directly calculates the chi2 from the
                    kinematic maps

        """
        self.logger.info(f"Using WeightSolver: {__class__.__name__}")
        if (not ignore_existing_weights) and self.weight_file_exists():
            self.logger.info(
                f"Reading NNLS solution from existing output {self.weight_file}."
            )
            results = ascii.read(self.weight_file)
            weights = results["weights"]
            chi2_tot = results.meta["chi2_tot"]
            chi2_kin = results.meta["chi2_kin"]
            chi2_kinmap = results.meta["chi2_kinmap"]
        else:
            fname_nn_orbmat = self.direc_with_ml + "nn_orbmat.out"
            # If legacy result files do not exist, run weight solving.
            check = (
                os.path.isfile(self.fname_nn_kinem)
                and os.path.isfile(self.fname_nn_nnls)
                and os.path.isfile(fname_nn_orbmat)
            )
            if ignore_existing_weights or not check:
                for f in [self.fname_nn_kinem, self.fname_nn_nnls, fname_nn_orbmat]:
                    if os.path.isfile(f):
                        os.remove(f)
                # set the current directory to the directory in which
                # the models are computed
                cur_dir = os.getcwd()
                os.chdir(self.direc_no_ml)
                cmdstr = self.write_executable_for_weight_solver()
                with open(cmdstr) as f:
                    for line in f:
                        i = line.find(">>")
                        if i >= 0:
                            j = line.find(".log")
                            logfile = line[i + 3 : j + 4]
                            break
                self.logger.info(
                    "Fitting orbit library to the kinematic "
                    + f"data: {logfile[: logfile.rindex('/')]}"
                )
                p = subprocess.run(
                    "bash " + cmdstr,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    shell=True,
                )
                # clean up decompressed files
                for f_name in [
                    f"datfil/orblib_{self.ml}.dat",
                    f"datfil/orblibbox_{self.ml}.dat",
                ]:
                    if os.path.isfile(f_name):
                        os.remove(f_name)
                log_file = f"Logfile: {self.direc_no_ml + logfile}."
                if not p.stdout.decode("UTF-8"):
                    self.logger.info(
                        f"...done, NNLS problem solved - {cmdstr} exit code {p.returncode}. {log_file}"
                    )
                else:
                    text = f"...failed! {cmdstr} exit code {p.returncode}. Message: {p.stdout.decode('UTF-8')}"
                    if p.returncode == 127:  # command not found
                        text += "Check DYNAMITE legacy_fortran executables."
                        self.logger.error(text)
                        os.chdir(cur_dir)
                        raise FileNotFoundError(text)
                    text += f"{log_file} Be wary: DYNAMITE may crash..."
                    self.logger.warning(text)
                    os.chdir(cur_dir)
                    raise RuntimeError(text)
                # set the current directory to the dynamite directory
                os.chdir(cur_dir)
                # delete existing .yaml files and copy current config file
                # into model directory
                self.config.copy_config_file(self.direc_with_ml)
            else:  # If legacy output files exist, just create the weight file
                self.logger.info(
                    "Reading NNLS solution from existing legacy output and converting to weights file."
                )
            # Now the legacy result files exist -> read, calculate
            # kinmapchi2, and save to the weight file.
            weights, chi2_tot, chi2_kin = self.get_weights_and_chi2_from_orbmat_file()
            chi2_kinmap = self.chi2_kinmap(weights)
            # save the output
            results = table.Table()
            results["weights"] = weights
            results.meta = {
                "chi2_tot": chi2_tot,
                "chi2_kin": chi2_kin,
                "chi2_kinmap": chi2_kinmap,
            }
            results.write(self.weight_file, format="ascii.ecsv", overwrite=True)
            # clean up
            if os.path.isfile(fname_nn_orbmat):
                os.remove(fname_nn_orbmat)
        return weights, chi2_tot, chi2_kin, chi2_kinmap

    def write_executable_for_weight_solver(self):
        """write executable bash script file

        Parameters
        ----------
        None

        Returns
        -------
        string
            the name of the bash script file to execute

        """
        nn = f"ml{self.ml:{self.sformat}}/nn"
        cmdstr = f"cmd_nnls_{self.ml}"
        txt_file = open(cmdstr, "w")
        txt_file.write("#!/bin/bash" + "\n")
        txt_file.write("# if the gzipped orbit library exist unzip it" + "\n")
        txt_file.write(
            f"test -e datfil/orblib_{self.ml}.dat || bunzip2 -c  datfil/orblib.dat.bz2 > datfil/orblib_{self.ml}.dat"
            + "\n"
        )
        txt_file.write(
            f"test -e datfil/orblibbox_{self.ml}.dat || bunzip2 -c  datfil/orblibbox.dat.bz2 > datfil/orblibbox_{self.ml}.dat"
            + "\n"
        )
        if self.system.is_bar_disk_system():
            txt_file.write(
                f"test -e {self.legacy_directory}/triaxnnls_bar"
                + f' || {{ echo "File {self.legacy_directory}/triaxnnls_bar not found." && exit 127; }}\n'
            )
            txt_file.write(
                "test -e "
                + str(nn)
                + "_kinem.out || "
                + self.legacy_directory
                + f"/triaxnnls_bar < {nn}.in >> {nn}ls.log "
                "|| exit 1\n"
            )
        elif self.CRcut is True:
            txt_file.write(
                f"test -e {self.legacy_directory}/triaxnnls_CRcut"
                + f' || {{ echo "File {self.legacy_directory}/triaxnnls_CRcut not found." && exit 127; }}\n'
            )
            txt_file.write(
                "test -e "
                + str(nn)
                + "_kinem.out || "
                + self.legacy_directory
                + f"/triaxnnls_CRcut < {nn}.in >> {nn}ls.log "
                "|| exit 1\n"
            )
        else:
            txt_file.write(
                f"test -e {self.legacy_directory}/triaxnnls_noCRcut"
                + f' || {{ echo "File {self.legacy_directory}/triaxnnls_noCRcut not found." && exit 127; }}\n'
            )
            txt_file.write(
                "test -e "
                + str(nn)
                + "_kinem.out || "
                + self.legacy_directory
                + f"/triaxnnls_noCRcut < {nn}.in >> {nn}ls.log "
                "|| exit 1\n"
            )
        txt_file.close()
        return cmdstr

    def read_weights(self):
        """Read ``nn_orb.out`` to astropy table

        this contains oribtal weights, orbit type, and other columns

        Returns
        -------
        None
            sets ``self.weights`` which is an astropy table containing
            the orbital weights

        """
        fname = self.direc_with_ml + "nn_orb.out"
        col_names = [
            "orb_idx",
            "E_idx",
            "I2_idx",
            "I3_idx",
            "totalnotregularizable",  # see line 535 of orblib_f.f90
            "orb_type",
            "weight",
            "lcut",
        ]  # lines 1321-1322 of triaxnnls_CRcut.f90
        # NOTE: column 'lcut' is not present if different "triaxnnls" file used
        dtype = [int, int, int, int, int, int, float, int]
        weights = np.genfromtxt(fname, skip_header=1, names=col_names, dtype=dtype)
        weights = table.Table(weights)
        self.weights = weights

    def read_nnls_orbmat_rhs_and_solution(self):
        """Read ``nn_orbmat.out``

        This contains the matrix and right-hand-side for the NNLS problem, and
        the solution

        Returns
        -------
        tuple
            (orbmat, rhs, solution)

        """
        fname = self.direc_with_ml + "nn_orbmat.out"
        orbmat_shape = np.loadtxt(fname, max_rows=1, dtype=int)
        orbmat_size = np.prod(orbmat_shape)
        tmp = np.loadtxt(fname, skiprows=1)
        orbmat = tmp[0:orbmat_size]
        orbmat = np.reshape(orbmat, orbmat_shape)
        orbmat = orbmat.T
        rhs = tmp[orbmat_size : orbmat_size + orbmat_shape[1]]
        solution = tmp[orbmat_size + orbmat_shape[1] :]
        return orbmat, rhs, solution

    def get_weights_and_chi2_from_orbmat_file(self):
        """
        Get weights and chi2 from ``nn_orbmat.out``

        **Note**: Chi2 values returned differ from `read_chi2` method.
        See that docstring for more.

        Returns
        -------
        tuple
            (weights, chi2_all, chi2_gh), where:

                -   weights : array of orbit weights
                -   chi2_all : sum of squared residuals for intrinsic masses,
                    projected_masses and GH coefficients h_1 to h_n
                -   chi2_kin : sum of squared residuals for GH coefficients h_1 to h_n

        """
        A, b, weights = self.read_nnls_orbmat_rhs_and_solution()
        chi2_vector = (np.dot(A, weights) - b) ** 2.0
        chi2_tot = np.sum(chi2_vector)

        stars = self.system.get_unique_triaxial_visible_component()
        if self.system.is_bar_disk_system():
            mge = stars.mge_lum + stars.disk_lum
        else:
            mge = stars.mge_lum

        intrinsic_masses = mge.get_intrinsic_masses_from_file(self.direc_no_ml)
        projected_masses = mge.get_projected_masses_from_file(self.direc_no_ml)
        n_intrinsic = np.prod(intrinsic_masses.shape)
        n_apertures = len(projected_masses)
        chi2_kin = np.sum(chi2_vector[1 + n_intrinsic + n_apertures :])
        return weights, chi2_tot, chi2_kin

    def read_chi2(self):
        """Read chi2 values from `nn_kinem.out`

        Taken from old `schwpy` code, lines 181-212 of schw_domoditer.py

        **Note**:
        This is a legacy method for reading legacy output and it not used by
        default. Instead we use ``self.get_chi2_from_orbmat`` get chi2 values.
        The chi2 value definitions of this method are NOT the same chi2 values
        given by ``self.get_chi2_from_orbmat``. They differ in
        (i) including intrinsic/projected mass constraints, and (ii) using
        h1/h2 vs V/sigma, and (iii) if CRcut==True, whether the 'cut' orbits
        - with artificially large h1 - are included (here they aren't)

        Returns
        -------
        tuple
            (chi2, kinchi2) where:
                -   chi2 = sum of sq. residuals of observed GH coefficients h_1
                    to h_N
                -   kinchi2 = sum of sq. residuals of V, sigma, and GH
                    coefficients from h_3 to h_N

        """
        # read amount of observables and kinematic moments
        fname = self.fname_nn_kinem
        a = self.__read_file_element(fname, [1, 1], [1, 2])
        ngh = np.int64(a[1])  # number of 'observables'
        nobs = np.int64(a[1])
        # nvel = np.int64(a[0])
        # ncon = np.int64(a[0])
        rows = 3 + np.arange(nobs)  # rows 1- 9
        cols = 3 + np.zeros(nobs, dtype=int)  # skip over text
        fname = self.fname_nn_nnls
        chi2vec = self.__read_file_element(fname, rows, cols)
        chi2vec = np.double(chi2vec)
        chi2 = sum(chi2vec)
        fname = self.fname_nn_kinem
        ka = np.genfromtxt(fname, skip_header=1)
        k = np.arange(ngh) * 3 + 3
        # k = is array of column indices, for [V, sigma, h3, ..., h_ngh]
        #                       observed   modelled        error
        kinchi2 = sum(sum(pow(((ka[:, k] - ka[:, k + 1]) / ka[:, k + 2]), 2.0)))
        return chi2, kinchi2

    def __read_file_element(self, infile, rows, cols):
        """Read fields in a tabular data according to the their row/column.

        Taken from schwpy schw_misc

        Parameters
        ----------
        infile : string
            input file
        rows : array of ints
            row array of locations indexed starts from 1
        cols : array of ints
            column array of locations indexed starts from 1

        Returns
        -------
        array read from given locations in file

        """
        lines = [line.rstrip("\n").split() for line in open(infile)]
        output = []
        for i in range(0, len(rows)):
            output.append(lines[rows[i] - 1][cols[i] - 1])
        return output


class NNLS(WeightSolver):
    """Python implementations of NNLS weight solving

    Uses either scipy.optimize.nnls or cvxopt as backends. This constructs the
    NNLS matrix and rhs, solves, and saves the result.

    Parameters
    ----------
    nnls_solver : string
        one of ``scipy``, ``cvxopt`` or ``adelie``. ``adelie`` uses
        coordinate-descent BVLS with an augmented Lagrangian on the total-mass
        constraint; see ``solve_adelie_alm``

    Weight solver settings
    ----------------------
    nnls_dtype : string, optional
        ``'float64'`` (default) or ``'float32'``. ``'float32'`` roughly
        halves the memory of the retained orbit-library data
        (``vel_histograms``/``intrinsic_masses``/``projected_masses``) and
        of the NNLS matrix/solve arrays, at the cost of a small precision
        loss. Validated against float64 on NGC6278 (matrix-only and
        end-to-end): chi2 agrees to <0.001%, KKT violation stays within the
        range of previously accepted float64 solutions. Not yet validated on
        datasets whose constraint-row scaling differs from NGC6278/omega
        Cen - carries the same caveat as ``adelie_mu`` (see
        ``solve_adelie_alm``). Applies to the orbit-library data and the
        constructed matrix regardless of ``nnls_solver``, but has only been
        validated with ``nnls_solver='adelie'``.

    """

    def __init__(self, nnls_solver=None, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(f"{__name__}.{__class__.__name__}")
        if nnls_solver is None:
            nnls_solver = self.settings["nnls_solver"]
        assert nnls_solver in ["scipy", "cvxopt", "adelie", "admm"], (
            "Unknown nnls_solver"
        )
        self.nnls_solver = nnls_solver
        # ALM settings for the adelie solver. On NGC6278, where scipy provides
        # a verified optimum, mu between 1e5 and 1e7 reproduces it; 1e7 gave the
        # lowest KKT violation (7e-11) and a monotone gap, whereas 1e6
        # oscillated. See docs/source/adelie_branch_migration.md.
        self.adelie_mu = float(self.settings.get("adelie_mu", 1.0e7))
        self.adelie_alm_iters = int(self.settings.get("adelie_alm_iters", 200))
        # adelie_tol / adelie_gap_tol are set below, AFTER nnls_dtype: their
        # safe values depend on the dtype's epsilon.
        # Coordinate-descent budget for ONE inner BVLS solve. adelie's own
        # default is 1e5; the 2e5 kept here preserves the previous hard-coded
        # value. Exposed as a setting because at omega Cen's matrix size a
        # single sweep costs seconds, so this is effectively a wall-clock
        # budget, and because the saturation path needs to be reachable in
        # tests without a pathological problem.
        self.adelie_bvls_max_iters = int(
            self.settings.get("adelie_bvls_max_iters", int(2e5))
        )
        # cvxopt interior-point tolerances (abstol/reltol/feastol share one
        # value) and iteration cap; the library defaults are 1e-7/1e-6/1e-7
        # and 100. 1e-11 is not conservatism: measured against an independent
        # constrained solve (scipy lsq_linear) on four seeds of a matrix with
        # this row structure, cvxopt lands 1.25-3.3x above the optimal chi2 at
        # 1e-9 and at 1.00-1.03x at 1e-11, for ~3 extra iterations. cvxopt
        # reports 'optimal' in BOTH cases, so the status flag will not warn
        # you. See dev_tests/test_cvxopt_equality.py.
        self.cvxopt_tol = float(self.settings.get("cvxopt_tol", 1e-11))
        self.cvxopt_maxiters = int(self.settings.get("cvxopt_maxiters", 200))
        # cvxopt's VENDORED OpenBLAS is built SINGLE-THREADED: measured
        # cvxopt.lapack.potrf at 0.06 Tflop/s at ANY thread count, vs
        # scipy's dpotrf (which links the system multithreaded LAPACK) at
        # 0.93-1.33 Tflop/s for p=12000-16000. At omega Cen's p=45000 that
        # is ~506 s per Cholesky through cvxopt's own KKT solver vs ~20 s
        # routed through scipy - and coneqp does one Cholesky per interior
        # point iteration. "custom" replaces cvxopt's KKT reduction with a
        # hand-written one (see CvxoptNonNegSolver.make_kktsolver) that
        # calls scipy's dpotrf/dpotrs instead; iterates are identical to
        # 1e-15 because it is the same linear algebra, just faster BLAS.
        # Measured speedup: 1.76x (p=2000) up to 5.95x (p=12000). It only
        # applies when there is exactly ONE equality row (our total-mass
        # constraint); with zero or more than one it silently falls back
        # to "default" and logs why.
        self.cvxopt_kktsolver = str(self.settings.get("cvxopt_kktsolver", "custom"))
        assert self.cvxopt_kktsolver in ("custom", "default"), (
            "cvxopt_kktsolver must be 'custom' or 'default'"
        )
        # cvxopt's own progress printout goes to stdout, not the logger, and
        # a multi-hour solve on stdout is easy to lose (piped to a file no
        # one is tailing, or swallowed by a batch scheduler). Default False
        # preserves today's silence on stdout; the INFO-level summary logged
        # after every solve (status/iterations/gap/wall time) and the
        # per-factorization heartbeat below are independent of this flag and
        # always happen, because a multi-hour solve that prints NOTHING is
        # indistinguishable from a hang - see the module docstring incident
        # where a silent 31.5h run finished 0/90 models.
        self.cvxopt_show_progress = bool(
            self.settings.get("cvxopt_show_progress", False)
        )
        # Heartbeat cadence for the custom kktsolver: log at INFO every Nth
        # factorization. At p=45000 one factorization is ~20s with the
        # custom solver, so logging every 1 is cheap and is the ONLY signal
        # that distinguishes "still working" from "stuck" during a long
        # solve, given cvxopt's own progress output is off by default.
        self.cvxopt_log_every = int(self.settings.get("cvxopt_log_every", 1))
        # ADMM (nnls_solver="admm"): fixed-rho splitting on the same Gram
        # problem, reusing the single-equality Schur complement above but
        # factoring ONCE (M = P + rho*I does not depend on the iterate,
        # unlike interior point's P + W^-2) and reusing that factorization
        # for every iteration. Measured at p=45000: factor 24.1s + 400
        # iterations in 499s = 8.72 min total, peak RSS 15.6 GB, vs cvxopt's
        # ~422 GB for the same problem. rho is a pure cost knob: it changes
        # iteration count by up to 3 orders of magnitude but the CONVERGED
        # answer is identical to 1e-15 across 4 decades of rho (verified).
        # Default None means "pick rho = trace(P)/p", a scale-matched guess
        # that is not tuned per-problem; pass a numeric value to override.
        self.admm_rho = self.settings.get("admm_rho", None)
        if self.admm_rho is not None:
            self.admm_rho = float(self.admm_rho)
        # ridge_lambda (cvxopt/admm, Gram paths): diagonal ridge applied to the
        # column-normalised P as  P += ridge_lambda * mean(diag(P))  BEFORE
        # solving - Vasiliev Eq. 7 regularisation, dimensionless so a value
        # measured at one p carries to another. chi2 continues to be reported
        # on the UNREGULARISED problem (raw-G quadratic form / C-identity
        # shift bookkeeping handle this automatically). Measured on the real
        # omega Cen problem at p=45000 (_diag_19_ridge_real, 2026-08-26):
        # lambda=10 is the smallest value where two independent solvers agree
        # on the WEIGHT VECTOR to better than 1e-6 (5.4e-07 there; 2.1e-06 at
        # lambda=3), and it collapses ADMM's iteration count from >4000
        # (unconverged cap) to 459. Default 0.0: no ridge, behaviour identical
        # to before this setting existed.
        self.ridge_lambda = float(self.settings.get("ridge_lambda", 0.0))
        # absolute per-diagonal shift the last _apply_diagonal_ridge added
        self._ridge_shift = 0.0
        self.admm_max_iters = int(self.settings.get("admm_max_iters", 4000))
        self.admm_tol = float(self.settings.get("admm_tol", 1e-11))
        # admm_free_p (gram_blockwise only): destroy P inside the solver by
        # factoring M = P + rho*I IN PLACE, then recover chi2's quadratic
        # form from the Cholesky factor alone (see AdmmNonNegSolver.
        # gram_quadratic_form). Cuts the solve's resident big matrices from
        # G+P+C to C. Default off: numerics are identical to rounding, but
        # the input P is consumed and the caller must not reuse it.
        self.admm_free_p = bool(self.settings.get("admm_free_p", False))
        if self.admm_free_p and not self.gram_blockwise:
            self.logger.warning(
                "admm_free_p=True has no effect without gram_blockwise=True; "
                "the materialized-A path needs A after the solve anyway."
            )
        # gram_blockwise (nnls_solver="cvxopt" or "admm" only): accumulate
        # P = A_rest^T A_rest / outer(col,col) and q one row block at a time
        # via NormalEquationAccumulator, instead of materializing A_rest
        # (371212 x 45000 = 124 GiB at float64 for omega Cen, with a second
        # full-size copy currently allocated on the way to P) and reducing
        # it. P itself is only ~15.1 GiB regardless of row count, so this is
        # the difference between a solve that peaks near 431 GB and one that
        # peaks near ~40 GB. Default False: nothing may change silently, and
        # the two paths are algebraically identical (not just "close") -
        # see dev_tests/test_gram_blockwise.py - so turning this on must
        # never change a reference result beyond rounding.
        self.gram_blockwise = bool(self.settings.get("gram_blockwise", False))
        assert self.gram_blockwise in (True, False)
        if self.gram_blockwise and nnls_solver not in ("cvxopt", "admm"):
            self.logger.warning(
                f"gram_blockwise=True has no effect for nnls_solver="
                f"'{nnls_solver}' (only 'cvxopt' and 'admm' use the Gram "
                "path); ignoring."
            )
        # Optional float32 mode: roughly halves the memory of the orbit
        # library data retained for the solve (vel_histograms/intrinsic_masses
        # /projected_masses) and of the NNLS matrix/solve arrays. Validated
        # against float64 on NGC6278 (matrix-only and end-to-end): chi2 agrees
        # to <0.001%, KKT violation stays within the range of previously
        # accepted float64 solutions. Not yet validated on datasets whose
        # constraint-row scaling differs from NGC6278/omega Cen - adelie_mu
        # itself carries the same caveat (see solve_adelie_alm docstring).
        nnls_dtype = self.settings.get("nnls_dtype", "float64")
        assert nnls_dtype in ("float32", "float64"), (
            "nnls_dtype must be 'float32' or 'float64'"
        )
        self.nnls_dtype = np.float32 if nnls_dtype == "float32" else np.float64
        # Solver tolerances are DERIVED FROM THE DTYPE, because a tolerance at
        # or below the dtype's epsilon makes the convergence test unsatisfiable:
        # the solve can then only ever stop by exhausting its iteration budget.
        # This is not hypothetical. A production grid inherited the float64
        # default 1e-10 while running float32 (eps 1.19e-07) and every BVLS call
        # ran to max_iters=2e5. At omega Cen's matrix size that is ~29 h per ALM
        # iterate, x200 iterates: five workers sat in ALM iterate 0 for 31.5 h
        # and finished 0 of 90 models, with flat RSS and nothing in the logs.
        _eps = np.finfo(self.nnls_dtype).eps
        _tol_default = 1.0e-10 if self.nnls_dtype is np.float64 else 1.0e-6
        # gap_tol likewise cannot beat the accuracy of the inner solve that
        # produces the weights it measures. Kept at the historical 1e-10 for
        # float64 so results do not shift; at float32 that is below eps and
        # therefore dead code, so it tracks the dtype instead.
        _gap_default = 1.0e-10 if self.nnls_dtype is np.float64 else 1.0e-6
        self.adelie_tol = float(self.settings.get("adelie_tol", _tol_default))
        self.adelie_gap_tol = float(self.settings.get("adelie_gap_tol", _gap_default))
        for _name, _val in (
            ("adelie_tol", self.adelie_tol),
            ("adelie_gap_tol", self.adelie_gap_tol),
        ):
            if _val <= _eps:
                self.logger.warning(
                    f"{_name}={_val:.1e} is at or below the {nnls_dtype} epsilon "
                    f"({_eps:.2e}), so the test it controls can never be "
                    "satisfied and the solver will run its full iteration "
                    f"budget every time. Raise {_name} above {_eps:.2e} or use "
                    "nnls_dtype 'float64'."
                )
        # Stream orbit-library histogram reads set-by-set in the fused adelie
        # constructor. Pure memory setting: results are bit-identical either
        # way (validated by dev_tests/_real_fused_check.py).
        self.stream_reads = bool(self.settings.get("stream_orblib_reads", False))
        self.get_observed_mass_constraints()

    def get_observed_mass_constraints(self):
        """Get aperture+intrinsic mass constraints from MGE

        Returns the projected masses of the MGE for the kinematic data
        apertures.

        Returns
        -------
        None
            sets attributes:

                - ``self.intrinsic_masses``
                - ``self.intrinsic_mass_error``
                - ``self.projected_masses``
                - ``self.projected_mass_error``
                - ``self.total_mass``
                - ``self.total_mass_error``
                -   constraint counts ``self.n_intrinsic``, ``self.n_apertures``
                    and ``self.n_mass_constraints``

        """
        if self.system.is_bar_disk_system():
            mge = self.system.get_unique_bar_component().mge_lum_tot
        else:
            mge = self.system.get_unique_triaxial_visible_component().mge_lum
        # intrinsic mass
        self.intrinsic_masses = mge.get_intrinsic_masses(self.model, nocalc=True)[1]
        self.intrinsic_mass_error = self.settings["lum_intr_rel_err"]
        # projected
        self.projected_masses = mge.get_projected_masses(nocalc=True)
        self.projected_mass_error = self.settings["sb_proj_rel_err"]
        # total mass constraint
        self.total_mass = np.sum(self.intrinsic_masses)
        self.total_mass_error = max(abs(1.0 - self.total_mass), 1e-8)
        # enumerate the mass constriants
        n_intrinsic = np.prod(self.intrinsic_masses.shape)
        n_apertures = len(self.projected_masses)
        self.n_intrinsic = n_intrinsic
        self.n_apertures = n_apertures
        # mass constraints = total mass (1) + intrinsic mass + aperture mass
        self.n_mass_constraints = 1 + n_intrinsic + n_apertures

    def construct_nnls_matrix_and_rhs(self, orblib):
        """construct nnls matrix_and rhs

        Parameters
        ----------
        orblib : ``dyn.orblib.OrbitLibrary``
            an orbit library

        Returns
        -------
        tuple
            (orbmat, rhs). ``orbmat`` has shape (n_constraints, n_orbs) and is
            **Fortran-ordered** - see the note below, and keep it that way.

        Notes
        -----
        This assembles the largest array in the run (125 GiB for omega Cen at
        45000 orbits), so it is written to touch that array as few times as
        possible: sizes are computed up front rather than grown by np.vstack,
        the memory order is chosen to match how the blocks arrive and how the
        solvers want to read them, and blocks are written through a reshaped
        view of the destination so no full-size temporary is ever materialised.
        Each of those is commented at its site.

        On the NGC5139 MUSE+HST test library (322403 x 3840, 9.9 GiB) this
        takes 3.9s, against 17.2s before those three changes, and produces a
        bit-identical matrix. dev_tests/_real_orblib_check.py reproduces both
        the timing and the comparison; dev_tests/test_nnls_matrix_assembly.py
        checks the assembly arithmetic without needing a stored orbit library.

        """
        # construct vector of observed constraints (con), errors (econ) and
        # matrix of orbit propertites (orbmat)
        dtype = self.nnls_dtype
        stars = self.system.get_unique_triaxial_visible_component()
        # Size con/econ/orbmat up front. The observed values depend only on the
        # kinematic data, not on the orbit library, so this pre-pass is cheap -
        # and it avoids growing orbmat by np.vstack per kinematic set, which
        # reallocates and copies the whole matrix each time (~125 GiB for
        # omega Cen). The results are kept and reused in the loop below.
        obs_values = [
            kins.get_observed_values_and_uncertainties(self.settings)
            for kins in stars.kinematic_data
        ]
        n_rows = self.n_mass_constraints + sum(np.size(v) for v, _ in obs_values)
        con = np.zeros(n_rows, dtype=dtype)
        econ = np.zeros(n_rows, dtype=dtype)
        # F order matters here. Each kinematic block arrives as
        # (n_orbs, n_constraints) and is written in transposed; against an
        # F-ordered destination that is a memcpy, against a C-ordered one it is
        # a strided shuffle of the whole matrix (measured 1.00s -> 0.08s per
        # 2.3 GiB block). It also hands the solvers the layout they want:
        # solve_adelie_alm's X is F-contiguous, so building it from A stops
        # being a full reorder (2.38s -> 0.33s). Only the chi2 matvec A @ w is
        # slightly slower (0.04s -> 0.07s), which is noise beside the rest.
        orbmat = np.zeros((n_rows, orblib.n_orbs), dtype=dtype, order="F")
        # total mass
        con[0] = self.total_mass
        econ[0] = self.total_mass_error
        orbmat[0, :] = 1.0
        # intrinsic mass
        idx = slice(1, 1 + self.n_intrinsic)
        con[idx] = np.ravel(self.intrinsic_masses)
        error = self.intrinsic_masses * self.intrinsic_mass_error
        error = np.abs(np.ravel(error))
        error[np.where(error <= 0.0)] = 1.0e-16
        econ[idx] = np.abs(np.ravel(error))
        orb_int_masses = orblib.intrinsic_masses
        orb_int_masses = np.reshape(orb_int_masses, (orblib.n_orbs, -1))
        orbmat[idx, :] = orb_int_masses.T
        # projected mass
        idx = slice(1 + self.n_intrinsic, 1 + self.n_intrinsic + self.n_apertures)
        con[idx] = self.projected_masses
        econ[idx] = np.abs(self.projected_masses * self.projected_mass_error)
        orbmat[idx, :] = np.hstack(orblib.projected_masses).T
        # add kinematics to con, econ, orbmat
        kins_and_orb_veldist = zip(
            stars.kinematic_data, orblib.vel_histograms, obs_values
        )
        idx_ap_start = 0
        idx_row = self.n_mass_constraints
        for kins, orb_veldist, tmp in kins_and_orb_veldist:
            # pick out the projected masses for this kinematic set
            n_ap = kins.n_spatial_bins  # OK for all kinematics
            idx_ap_end = idx_ap_start + n_ap
            prj_mass_i = self.projected_masses[idx_ap_start:idx_ap_end]
            idx_ap_start += n_ap
            obs_kins, obs_kins_err, orb_kins = self._prepare_kinematic_block(
                kins, orb_veldist, tmp, prj_mass_i
            )
            # append constraints/errors/orbits to con/econ/orbmat
            n_orb_constraints = orb_kins.size // orblib.n_orbs
            idx_row_end = idx_row + obs_kins.size
            assert n_orb_constraints == obs_kins.size, (
                f"{type(kins).__name__}: orbit library gives "
                f"{n_orb_constraints} constraints per orbit but the "
                f"kinematic data gives {obs_kins.size}"
            )
            # slice assignment casts to dtype in place, no extra full copy
            con[idx_row:idx_row_end] = obs_kins
            econ[idx_row:idx_row_end] = np.ravel(obs_kins_err)
            # orb_kins comes back from transform_orblib_to_observables as a
            # transposed/moveaxis'd view, so np.reshape(orb_kins, (n_orbs, -1))
            # cannot be a view and copies the whole block - 2.1s and a full
            # extra copy of the matrix on NGC5139, so ~100 GiB for omega Cen.
            # Reshaping the *destination* is a view instead, and the write then
            # goes straight from orb_kins with no temporary at all.
            #
            # ndarray.reshape() silently returns a COPY when it cannot return a
            # view. As an assignment target that is a silent no-op: the block
            # would be left as zeros with no error raised anywhere. The assert
            # is what makes that failure loud, so do not drop it. (The .shape=
            # setter raises natively, but is deprecated in numpy 2.5 and we
            # support numpy>=1.26.)
            dest = orbmat[idx_row:idx_row_end, :].T.reshape(orb_kins.shape)
            assert np.shares_memory(dest, orbmat), (
                "orbmat block write got a copy, not a view - block would be silently left at zero"
            )
            dest[...] = orb_kins
            idx_row = idx_row_end
        # divide constraint vector and matrix by errors
        if np.any(con[econ == 0] != 0):
            txt = "Weight solving fail: zero errors for nonzero constraints!"
            self.logger.error(txt)
            raise ValueError(txt)
        # previous statement: rhs = con/econ, np.divide has the "where" clause
        rhs = np.zeros_like(con)
        np.divide(con, econ, out=rhs, where=econ != 0)  # con = econ = 0 is ok
        if np.any(np.ravel(orbmat[econ == 0]) != 0):
            err_loc = np.nonzero(((orbmat != 0).T * (econ == 0)).T)
            txt = (
                f"Weight solving problem in {self.direc_with_ml}: "
                "zero errors for nonzero matrix coefficients at "
                f"[constraint no, orbit no] = {err_loc}! Matrix value(s) "
                f"there ({orbmat[err_loc]}) will be considered zero."
            )
            self.logger.warning(txt)
            orbmat[err_loc] = 0
        # previous statement: orbmat = (orbmat.T/econ).T, np.divide has "where"
        orbmat = orbmat.T
        np.divide(orbmat, econ, out=orbmat, where=econ != 0)
        return orbmat.T, rhs

    def construct_adelie_matrix_and_rhs(self, orblib):
        """Assemble adelie's augmented matrix directly — A never exists.

        Writes the sqrt(mu) penalty row into X row 0, streams every
        constraint block straight into rows 1.., divides by econ, then
        finishes with the SAME col_norm/divide/y steps as
        _build_augmented_X, so X/col_norm/y are bit-identical to the
        two-step construction used by the scipy/cvxopt paths. Saves one full
        matrix of RAM (~125 GiB for omega Cen), which is what lets several
        weight solves share a node.

        Returns an :class:`AdelieProblem`.
        """
        dtype = self.nnls_dtype
        stars = self.system.get_unique_triaxial_visible_component()
        # observed values depend only on the kinematic data; same pre-pass,
        # kept and reused, as in construct_nnls_matrix_and_rhs
        obs_values = [
            kins.get_observed_values_and_uncertainties(self.settings)
            for kins in stars.kinematic_data
        ]
        n_rows = self.n_mass_constraints + sum(np.size(v) for v, _ in obs_values)
        sqrt_mu = np.sqrt(self.adelie_mu)
        con = np.zeros(n_rows, dtype=dtype)
        econ = np.zeros(n_rows, dtype=dtype)

        if self.stream_reads:
            # Stream one kinematic set at a time: read -> transform -> write
            # its block into X -> free, so only that set's histograms and the
            # accumulated X are ever co-resident.
            orblib.read_vel_histograms(kin_sets=[0], skip_density=False)
            _downcast_orblib(orblib, self.nnls_dtype)
        n_orbs = orblib.n_orbs
        X = np.empty((n_rows, n_orbs), dtype=dtype, order="F")
        # row 0 is the ALM penalty row; it REPLACES A's total-mass row
        X[0, :] = sqrt_mu
        con[0] = self.total_mass
        econ[0] = self.total_mass_error
        # intrinsic mass -> X rows 1..n_intrinsic
        idx = slice(1, 1 + self.n_intrinsic)
        con[idx] = np.ravel(self.intrinsic_masses)
        error = np.abs(np.ravel(self.intrinsic_masses * self.intrinsic_mass_error))
        error[np.where(error <= 0.0)] = 1.0e-16
        econ[idx] = error
        X[idx, :] = np.reshape(orblib.intrinsic_masses, (n_orbs, -1)).T
        # projected mass: the constraint vector is known up front, the matrix
        # rows are filled per set inside the loop below (each set owns a
        # contiguous aperture range, so this is the same content the old
        # np.hstack(projected_masses).T produced, without the full-size
        # temporary it needed to build it).
        idx_prj = slice(1 + self.n_intrinsic, 1 + self.n_intrinsic + self.n_apertures)
        con[idx_prj] = self.projected_masses
        econ[idx_prj] = np.abs(self.projected_masses * self.projected_mass_error)
        # kinematics: identical block sequence in both read modes. Rows are
        # independent, so streaming produces a bit-identical X.
        idx_ap_start = 0
        idx_row = self.n_mass_constraints
        for si, kins in enumerate(stars.kinematic_data):
            if self.stream_reads and si > 0:
                orblib.read_vel_histograms(kin_sets=[si], skip_density=True)
                _downcast_orblib(orblib, self.nnls_dtype)
            orb_veldist = orblib.vel_histograms[si]
            assert orb_veldist is not None, f"no histogram for kinematic set {si}"
            n_ap = kins.n_spatial_bins
            prj_mass_i = self.projected_masses[idx_ap_start : idx_ap_start + n_ap]
            prj_parts_i = orblib.projected_masses[si]
            assert prj_parts_i is not None, (
                f"no projected masses for kinematic set {si}"
            )
            X[
                1 + self.n_intrinsic + idx_ap_start : 1
                + self.n_intrinsic
                + idx_ap_start
                + n_ap,
                :,
            ] = prj_parts_i.T
            obs_kins, obs_kins_err, orb_kins = self._prepare_kinematic_block(
                kins, orb_veldist, obs_values[si], prj_mass_i
            )
            n_orb_constraints = orb_kins.size // n_orbs
            idx_row_end = idx_row + obs_kins.size
            assert n_orb_constraints == obs_kins.size, (
                f"{type(kins).__name__}: orbit library gives "
                f"{n_orb_constraints} constraints per orbit but the "
                f"kinematic data gives {obs_kins.size}"
            )
            con[idx_row:idx_row_end] = obs_kins
            econ[idx_row:idx_row_end] = obs_kins_err
            dest = X[idx_row:idx_row_end, :].T.reshape(orb_kins.shape)
            assert np.shares_memory(dest, X), (
                "X block write got a copy, not a view - block would be silently left at zero"
            )
            dest[...] = orb_kins
            idx_row = idx_row_end
            idx_ap_start += n_ap
            if self.stream_reads:
                # free this set's histograms before reading the next one;
                # glibc returns these mmap-backed pages immediately
                orblib.vel_histograms[si] = None
        # guards: mirror the stock constructor. Row 0 has no econ semantics
        # (it is the penalty row); A's total-mass row becomes row0_vec below.
        if np.any(con[econ == 0] != 0):
            txt = "Weight solving fail: zero errors for nonzero constraints!"
            self.logger.error(txt)
            raise ValueError(txt)
        rhs = np.zeros_like(con)
        np.divide(con, econ, out=rhs, where=econ != 0)  # con = econ = 0 ok
        econ_body = econ[1:]
        # Only zero-error rows can offend, and normally there are none. Restrict
        # to those rows instead of building a (n_rows-1, n_orbs) bool mask: at
        # production scale that mask is ~16 GiB, and the `&` allocates a second
        # one before the first is freed - ~31 GiB transient at the exact moment
        # assembly is already at its RSS peak. The stock constructor above gets
        # this for free by masking rows first; keep the fused path equivalent.
        bad = np.nonzero(econ_body == 0)[0]
        if bad.size:
            blk = X[1:, :][bad]  # (n_bad, n_orbs), n_bad is normally 0
            nz_rows, nz_cols = np.nonzero(blk)
            if nz_rows.size:
                rr = bad[nz_rows]
                txt = (
                    f"Weight solving problem in {self.direc_with_ml}: "
                    "zero errors for nonzero matrix coefficients at "
                    f"[constraint no, orbit no] = {(rr + 1, nz_cols)}! Matrix "
                    f"value(s) there ({blk[nz_rows, nz_cols]}) will be "
                    "considered zero."
                )
                self.logger.warning(txt)
                X[1 + rr, nz_cols] = 0
        # divide rows by their errors: the same elementwise op as the stock
        # transposed-view divide, restricted to rows 1.. (row 0 of A becomes
        # row0_vec below, divided elementwise by econ[0] exactly as stock's
        # broadcast divide did to it).
        body = X[1:, :].T  # view (n_orbs, n_rows-1)
        np.divide(body, econ_body, out=body, where=econ_body != 0)
        col_norm, y = _scale_columns(X, rhs[1:], dtype)
        row0_vec = np.full(n_orbs, 1.0, dtype=dtype) / econ[0]
        return AdelieProblem(
            X=X, col_norm=col_norm, y=y, row0_vec=row0_vec, b0=float(rhs[0])
        )

    def _econ_divide_block(self, block, econ_slice, row_offset):
        """Guard + divide one constraint block by its own econ slice, IN
        PLACE, and return it.

        Mirrors the two guards applied to the full matrix in
        ``construct_nnls_matrix_and_rhs`` (around line 1041) and
        ``construct_adelie_matrix_and_rhs`` (around line 1153): zero-error
        rows with nonzero constraints are handled by the caller (checked
        once, up front, on the cheap ``con``/``econ`` vectors before any
        block exists); zero-error rows with nonzero matrix coefficients are
        zeroed and a warning logged. Applied per block instead of once after
        a full-size loop, because on this path there never IS a full-size
        matrix to mask in one shot — each block is discarded once it has
        been folded into the accumulator.

        THIS MUST RUN BEFORE THE BLOCK IS HANDED TO THE ACCUMULATOR. See the
        "ordering trap" note in :meth:`construct_gram_and_rhs_blockwise`.

        Parameters
        ----------
        block : array (rows, n_orbs)
            Modified in place.
        econ_slice : array (rows,)
        row_offset : int
            1-based row index of ``block[0]`` in the full constraint
            ordering (row 0 is the total-mass row, handled separately and
            never passed here), used only for the warning message.

        Returns
        -------
        array
            ``block``, divided by ``econ_slice`` (rows with zero econ are
            left as-is, having just been zeroed by the guard above).
        """
        bad = np.nonzero(econ_slice == 0)[0]
        if bad.size:
            nz_rows, nz_cols = np.nonzero(block[bad])
            if nz_rows.size:
                rr = bad[nz_rows]
                txt = (
                    f"Weight solving problem in {self.direc_with_ml}: "
                    "zero errors for nonzero matrix coefficients at "
                    f"[constraint no, orbit no] = {(rr + row_offset, nz_cols)}! "
                    f"Matrix value(s) there ({block[rr, nz_cols]}) will be "
                    "considered zero."
                )
                self.logger.warning(txt)
                block[rr, nz_cols] = 0
        np.divide(
            block, econ_slice[:, None], out=block, where=(econ_slice != 0)[:, None]
        )
        return block

    def construct_gram_and_rhs_blockwise(self, orblib):
        """Assemble cvxopt/ADMM's normal equations ``P``, ``q`` directly
        from the orbit library, one constraint block at a time — ``A``
        never exists.

        Mirrors the streaming loop in :meth:`construct_adelie_matrix_and_rhs`,
        but instead of writing each block into a ``(n_rows, n_orbs)`` matrix
        ``X``, it divides each block by ITS OWN econ slice and feeds it
        straight to a :class:`NormalEquationAccumulator`, which folds it
        into the running Gram matrix ``G = A[1:]^T A[1:]`` and cross term
        ``v = A[1:]^T b[1:]`` via one ``dsyrk`` call per block and then
        discards the block. ``A`` (371212 x 45000, 124 GiB at float64 for
        omega Cen) never exists; the Gram matrix ``G``/``P`` (45000^2,
        15.1 GiB) is the largest thing that does, regardless of how many
        constraint rows there are.

        THE ONE ORDERING TRAP. ``construct_nnls_matrix_and_rhs`` and
        ``construct_adelie_matrix_and_rhs`` both apply econ to ALL body rows
        AT ONCE, AFTER their loop finishes (they can do this because they
        hold the whole matrix). That shortcut is NOT available here: once a
        block is folded into ``G`` by ``dsyrk``, it cannot be rescaled row
        by row afterwards — the information needed to undo a wrong scaling
        is gone. Every block below is therefore divided by its own econ
        slice (via :meth:`_econ_divide_block`) BEFORE ``acc.add()`` is
        called, never after. Getting this wrong does not raise: it silently
        produces a plausible but WRONG ``P``. See
        ``dev_tests/test_gram_blockwise.py::test_econ_before_not_after_accumulation``,
        which is pinned specifically to fail if this ordering is violated.

        Row 0 (total mass) is skipped from the accumulator entirely — as in
        the classic cvxopt/admm branches, it becomes the equality
        constraint (``row0_vec``, ``b0``) rather than a QP row, worth about
        5 orders of magnitude of conditioning.

        The mass-constraint rows (intrinsic + projected masses) are ALSO
        kept as a small materialized array ``A_mass`` — their row count is
        tiny next to the kinematic rows — purely so ``chi2_kin`` can be
        split out of ``chi2_tot`` after solving without a second
        ``n_orbs x n_orbs`` Gram matrix: ``chi2_kin = chi2_rest -
        chi2_mass``, where ``chi2_rest`` (mass + kinematic rows together)
        comes from the single accumulated ``(G, v, b_sq_sum)`` via the
        quadratic form ``w'Gw - 2 w'v + ||b_rest||^2`` (no pass over ``A``
        needed), and ``chi2_mass = ||A_mass @ w - b_mass||^2`` is a cheap
        direct matvec since ``A_mass`` is small.

        Returns a :class:`GramProblem`.
        """
        dtype = self.nnls_dtype
        stars = self.system.get_unique_triaxial_visible_component()
        obs_values = [
            kins.get_observed_values_and_uncertainties(self.settings)
            for kins in stars.kinematic_data
        ]
        n_rows = self.n_mass_constraints + sum(np.size(v) for v, _ in obs_values)
        con = np.zeros(n_rows, dtype=np.float64)
        econ = np.zeros(n_rows, dtype=np.float64)

        if self.stream_reads:
            orblib.read_vel_histograms(kin_sets=[0], skip_density=False)
            _downcast_orblib(orblib, self.nnls_dtype)
        n_orbs = orblib.n_orbs

        con[0] = self.total_mass
        econ[0] = self.total_mass_error
        idx = slice(1, 1 + self.n_intrinsic)
        con[idx] = np.ravel(self.intrinsic_masses)
        error = np.abs(np.ravel(self.intrinsic_masses * self.intrinsic_mass_error))
        error[error <= 0.0] = 1.0e-16
        econ[idx] = error
        idx_prj = slice(1 + self.n_intrinsic, 1 + self.n_intrinsic + self.n_apertures)
        con[idx_prj] = self.projected_masses
        econ[idx_prj] = np.abs(self.projected_masses * self.projected_mass_error)
        # kinematics con/econ are filled per set inside the loop below, same
        # as the fused adelie constructor; con/econ themselves are cheap
        # (n_rows floats), only the matrix is the expensive part avoided here

        # guard #1: mirrors the check in construct_nnls_matrix_and_rhs /
        # construct_adelie_matrix_and_rhs. Needs con/econ for the KINEMATIC
        # rows too, so it must run after the loop fills them in - moved to
        # just before the accumulator is finalized, below.

        acc = NormalEquationAccumulator(n_orbs, dtype=np.float64)
        n_mass_rest = self.n_intrinsic + self.n_apertures
        A_mass = np.empty((n_mass_rest, n_orbs), dtype=np.float64, order="C")

        # intrinsic masses -> mass rows [0, n_intrinsic)
        econ_intrinsic = econ[idx]
        orb_int = np.reshape(orblib.intrinsic_masses, (n_orbs, -1)).T.astype(
            np.float64, copy=True
        )  # (n_intrinsic, n_orbs)
        self._econ_divide_block(orb_int, econ_intrinsic, row_offset=1)
        A_mass[0 : self.n_intrinsic, :] = orb_int
        con_intrinsic = con[idx]
        b_intrinsic = np.zeros(self.n_intrinsic, dtype=np.float64)
        np.divide(
            con_intrinsic, econ_intrinsic, out=b_intrinsic, where=econ_intrinsic != 0
        )
        acc.add(orb_int, b_intrinsic)
        del orb_int

        # projected masses + kinematics, per set (projected masses share the
        # set's contiguous aperture range at offset idx_ap_start, exactly as
        # in construct_adelie_matrix_and_rhs)
        idx_ap_start = 0
        idx_row = self.n_mass_constraints
        for si, kins in enumerate(stars.kinematic_data):
            if self.stream_reads and si > 0:
                orblib.read_vel_histograms(kin_sets=[si], skip_density=True)
                _downcast_orblib(orblib, self.nnls_dtype)
            orb_veldist = orblib.vel_histograms[si]
            assert orb_veldist is not None, f"no histogram for kinematic set {si}"
            n_ap = kins.n_spatial_bins
            prj_mass_i = self.projected_masses[idx_ap_start : idx_ap_start + n_ap]
            prj_parts_i = orblib.projected_masses[si]
            assert prj_parts_i is not None, (
                f"no projected masses for kinematic set {si}"
            )

            prj_block = prj_parts_i.T.astype(np.float64, copy=True)  # (n_ap, n_orbs)
            row0_prj = 1 + self.n_intrinsic + idx_ap_start
            econ_prj_i = econ[row0_prj : row0_prj + n_ap]
            con_prj_i = con[row0_prj : row0_prj + n_ap]
            self._econ_divide_block(prj_block, econ_prj_i, row_offset=row0_prj)
            mass_lo = self.n_intrinsic + idx_ap_start
            A_mass[mass_lo : mass_lo + n_ap, :] = prj_block
            b_prj_i = np.zeros(n_ap, dtype=np.float64)
            np.divide(con_prj_i, econ_prj_i, out=b_prj_i, where=econ_prj_i != 0)
            acc.add(prj_block, b_prj_i)
            del prj_block

            obs_kins, obs_kins_err, orb_kins = self._prepare_kinematic_block(
                kins, orb_veldist, obs_values[si], prj_mass_i
            )
            n_orb_constraints = orb_kins.size // n_orbs
            idx_row_end = idx_row + obs_kins.size
            assert n_orb_constraints == obs_kins.size, (
                f"{type(kins).__name__}: orbit library gives "
                f"{n_orb_constraints} constraints per orbit but the "
                f"kinematic data gives {obs_kins.size}"
            )
            con[idx_row:idx_row_end] = obs_kins
            econ[idx_row:idx_row_end] = obs_kins_err
            # same reshape identity as the fused adelie constructor's
            # X[idx_row:idx_row_end,:].T.reshape(orb_kins.shape) = orb_kins,
            # solved for the (rows, n_orbs) block directly: kin_block =
            # orb_kins.reshape(n_orbs, -1).T. Unlike the fused constructor,
            # this does NOT need to be a view into anything - the block is
            # fed to the accumulator and discarded, never written back.
            kin_block = orb_kins.reshape(n_orbs, -1).T.astype(np.float64, copy=True)
            econ_kin_i = econ[idx_row:idx_row_end]
            con_kin_i = con[idx_row:idx_row_end]
            self._econ_divide_block(kin_block, econ_kin_i, row_offset=idx_row)
            b_kin_i = np.zeros(obs_kins.size, dtype=np.float64)
            np.divide(con_kin_i, econ_kin_i, out=b_kin_i, where=econ_kin_i != 0)
            acc.add(kin_block, b_kin_i)
            del kin_block, orb_kins

            idx_row = idx_row_end
            idx_ap_start += n_ap
            if self.stream_reads:
                orblib.vel_histograms[si] = None

        # guard #1, now that con/econ are fully populated: zero-error rows
        # with nonzero constraints must never happen (guard #2, the
        # per-block matrix-coefficient check, already ran inside the loop
        # above via _econ_divide_block).
        if np.any(con[econ == 0] != 0):
            txt = "Weight solving fail: zero errors for nonzero constraints!"
            self.logger.error(txt)
            raise ValueError(txt)

        row0_vec = np.full(n_orbs, 1.0, dtype=np.float64) / econ[0]
        b0 = con[0] / econ[0]
        b_mass = np.concatenate(
            [b_intrinsic, np.zeros(self.n_apertures, dtype=np.float64)]
        )
        # fill in the projected-mass part of b_mass (was computed per set
        # above but not retained outside the loop; recompute cheaply here -
        # con/econ for those rows are tiny vectors, not the matrix)
        np.divide(
            con[idx_prj],
            econ[idx_prj],
            out=b_mass[self.n_intrinsic :],
            where=econ[idx_prj] != 0,
        )

        P, q, col_norm, b_max = acc.finalize()
        return GramProblem(
            P=P,
            q=q,
            col_norm=col_norm,
            b_max=b_max,
            G=acc.G,
            v=acc.v,
            b_sq_rest=acc.b_sq_sum,
            A_mass=A_mass,
            b_mass=b_mass,
            row0_vec=row0_vec,
            b0=float(b0),
        )

    def _prepare_kinematic_block(self, kins, orb_veldist, tmp, prj_mass_i):
        """Shared by all NNLS constructors.

        Scales the observed kinematics and their errors by this set's
        projected masses, zeroes the first and last point of 1D orbit
        velocity histograms IN PLACE (mimicking ``triaxnnls_CRcut.f90``),
        transforms the orbit library into the observed parameterisation,
        and applies the CRcut if enabled (only has an effect for
        GaussHermite). Returns flat ``(obs_kins, obs_kins_err, orb_kins)``.

        The zeroing is undone before returning: the two edge slices are
        (n_orbs x n_apertures each - tiny), and handing the library back
        exactly as it was received lets callers reuse it instead of paying
        for a second full read, which is by far the most expensive operation
        in a model.
        """
        hist_dim = len(orb_veldist.y[0, ..., 0].shape)  # 1D or 2D vel hists
        obs_kins, obs_kins_err = tmp
        obs_kins = (obs_kins.T * prj_mass_i).T
        obs_kins_err = (obs_kins_err.T * prj_mass_i).T
        edges_restore = []
        if hist_dim == 1:  # Do we need this for proper motions (2d hists)?
            edges_restore.append(
                (
                    orb_veldist.y,
                    orb_veldist.y[:, 0, :].copy(),
                    orb_veldist.y[:, -1, :].copy(),
                )
            )
            orb_veldist.y[:, 0, :] = 0.0
            orb_veldist.y[:, -1, :] = 0.0
        # transform orblib to same parameterisation as observed kinematics
        orb_kins = kins.transform_orblib_to_observables(orb_veldist, self.settings)
        if self.CRcut:
            orb_kins = self.apply_CR_cut(kins, orb_veldist, orb_kins)
        # put the zeroed velocity-histogram edges back, so orblib is left
        # exactly as it was handed to us (see the note above)
        for hist_y, first, last in edges_restore:
            hist_y[:, 0, :] = first
            hist_y[:, -1, :] = last
        return np.ravel(obs_kins), np.ravel(obs_kins_err), orb_kins

    def apply_CR_cut(self, kins, orb_losvd, orb_gh):
        r"""apply `CRcut`

        to solve the `counter rotating orbit problem`. This cuts orbits which
        have :math:`|V - V_\mathrm{obs}|> 3\sigma_\mathrm{obs}`. See
        Zhu+2018 MNRAS 2018 473 3000 for details

        Parameters
        ----------
        kins : a ``dyn.kinematics.Kinematic`` object
        orb_losvd : ``dyn.kinematics.Histogram``
            historgram of orblib losvds
        orb_gh : array
            array of input gh expansion coefficients, before the CRcut

        Returns
        -------
        array
            array of input gh expansion coefficients, after the CRcut

        """
        if type(kins) is not dyn_kin.GaussHermite:
            return orb_gh
        orb_mu_v = orb_losvd.get_mean()
        kins_data = kins.get_data()
        obs_mu_v = kins_data["v"]
        obs_sig_v = kins_data["sigma"]
        delta_v = np.abs(orb_mu_v - obs_mu_v)
        condition1 = np.abs(obs_mu_v) / obs_sig_v > 1.5
        condition2 = delta_v / obs_sig_v > 3.0
        condition3 = obs_mu_v * orb_mu_v < 0
        idx_cut = np.where(condition1 & condition2 & condition3)
        cut = np.zeros_like(orb_mu_v, dtype=bool)
        cut[idx_cut] = True
        naperture_cut = np.sum(cut, 1)
        # orbit 'j' is "bad" in naperture_cut[j] apertures
        # if an orbit is bad in 0 or 1 apertures, then we ignore this
        cut[naperture_cut < 1, :] = False
        # to cut an orbit, replace it's h1 by 3.0/dvhist(i)
        idx_cut = np.where(cut)
        dvhist = kins.hist_width / kins.hist_bins
        dvhist = np.max(dvhist)
        orb_gh[idx_cut[0], idx_cut[1], 0] = 3.0 / dvhist
        return orb_gh

    @staticmethod
    def kkt_violation(A, b, weights):
        r"""Optimality certificate for the NNLS solution.

        NNLS is convex, so the Karush-Kuhn-Tucker conditions are necessary
        **and sufficient**: a point satisfying them is a global optimum. With
        :math:`g = A^T(Aw-b)` they read, per orbit,

        - :math:`w_j > 0 \Rightarrow g_j = 0`
        - :math:`w_j = 0 \Rightarrow g_j \geq 0`   (adding weight cannot help)

        Two numbers are returned: the largest violation itself, and that
        violation scaled by ``||A_col|| * ||r||`` where ``r = Aw - b`` is the
        residual. By Cauchy-Schwarz :math:`|g_j| \leq \|A_{\cdot j}\|\,\|r\|`,
        so the scaled value lies in [0, 1] and measures how strongly the worst
        orbit is still aligned with the residual.

        The residual, not ``b``, is the correct denominator. Scaling by
        ``||b||`` was tried first and is unusable here: ``b[0] = 1e8`` makes the
        factor ~1e16, so a raw violation of 3.9e3 was reported as 3.9e-13 and a
        clearly unconverged solution looked optimal. That is the same failure
        mode as adelie's ``y_var``. ``||r||`` is a property of the solution and
        shrinks as it converges, which is what a convergence measure requires.

        Measured on solutions whose status is known independently:

        =========================== ========= ========== ==========
        solution                    chi2      raw        scaled
        =========================== ========= ========== ==========
        synthetic, exact            1.4e-32   1.1e-16    9.0e-09
        synthetic, scipy exact      9.2e-30   8.5e-15    2.8e-08
        synthetic, ALM unconverged  8.5e+00   4.0e+03    1.4e-05
        omega Cen, stored scipy     9.1e+09   7.1e+16    1.0e+00
        omega Cen, ALM              1.2e+04   4.5e+13    5.7e-03
        =========================== ========= ========== ==========

        The omega Cen scipy solution scores 1.0, the maximum: 27568 of its
        36000 orbits sit at zero while their gradients are negative, yet scipy
        reported convergence. chi2 cannot detect this, since comparing two
        solutions does not establish that either is optimal.

        Note what the table also shows: omega Cen's good solution (5.7e-03)
        scores WORSE than the synthetic unconverged one (1.4e-05). Being
        bounded in [0, 1] does not make the value comparable across problems,
        because what counts as small depends on the conditioning. Use it to
        rank solutions of one problem, and to catch gross failures near 1; do
        not read a fixed threshold as a convergence criterion.

        Parameters
        ----------
        A : array (n_constraints, n_orbits)
        b : array (n_constraints,)
        weights : array (n_orbits,)

        Returns
        -------
        tuple of float
            ``(scaled, raw)``. ``scaled`` is in [0, 1] and is comparable across
            problems; ``raw`` is dimensional and comparable only within one.
            Report both.

        References
        ----------
        Boyd, S. & Vandenberghe, L. 2004, Convex Optimization (Cambridge Univ.
            Press), Sect. 5.5.3 -- KKT conditions are sufficient, not only
            necessary, for a convex problem such as this one

        """
        resid = A @ weights - b
        grad = A.T @ resid
        viol = np.where(weights > 0, np.abs(grad), np.maximum(-grad, 0.0))
        raw = float(np.max(viol))
        # Cauchy-Schwarz denominator: |g_j| <= ||A_.j|| ||r||, so this is in
        # [0, 1]. An exactly-fitting solution has ||r|| -> 0 and the ratio is
        # then 0/0; guard it and report 0, which is the correct verdict.
        scale = np.linalg.norm(A, axis=0) * np.linalg.norm(resid)
        if not np.any(scale > 0):
            return 0.0, raw
        scale = np.where(scale > 0, scale, np.inf)
        return float(np.max(viol / scale)), raw

    @staticmethod
    def _build_augmented_X(A_rest, b_rest, sqrt_mu, dtype):
        """Build adelie's augmented, column-scaled matrix from A's body.

        Unit-L2 column scaling is an exact change of variable, since positive
        diagonal scaling preserves w >= 0, and the column norms span about 15
        orders of magnitude. The array is F-contiguous because coordinate
        descent accesses one column at a time.

        Bitwise identical to the inline construction this was extracted from
        (solve_adelie_alm, pre-2026-08-21); dev_tests/test_augmented_build.py
        pins that. Kept as a named step so the fused constructor
        (construct_adelie_matrix_and_rhs) shares the same finishing moves."""
        n_orbs = A_rest.shape[1]
        # Filled in place into an F-ordered buffer: np.vstack + `X / col_norm`
        # + np.asfortranarray would each allocate a full copy of X, so the
        # naive version peaks at 4x the matrix (~500 GiB for omega Cen).
        X = np.empty((A_rest.shape[0] + 1, n_orbs), dtype=dtype, order="F")
        X[0, :] = sqrt_mu
        X[1:, :] = A_rest
        col_norm, y = _scale_columns(X, b_rest, dtype)
        return X, col_norm, y

    @staticmethod
    def kkt_violation_augmented(
        row0_vec, b0, X_scaled, col_norm, resid_full, weights, mu
    ):
        r"""Optimality certificate computed from the augmented matrix alone.

        Same semantics as :meth:`kkt_violation` — returns ``(scaled, raw)``
        with scaled in [0, 1] — but expressed through the fused augmented
        matrix so A never has to exist:

            A[1:, j]   = col_norm[j] * X_scaled[1:, j]
            grad[j]    = row0_vec[j]*r0 + col_norm[j]*(X_scaled[1:]^T r)[j]
            ||A_.j||^2 = row0_vec[j]^2 + col_norm[j]^2 - mu

        where ``r`` is the plain residual ``Aw - b`` aligned to A's rows
        (slot 0 = the total-mass row) and col_norm includes the penalty row.
        Two passes over X instead of three over A.

        Algebraically identical to kkt_violation(A, b, w); differs in rounding
        only. dev_tests/test_surrogate_chi2_kkt.py checks agreement to rtol
        1e-10 and the exact-fit / degenerate-column guards.
        """
        r0 = float(row0_vec @ weights) - float(b0)
        r_rest = np.asarray(resid_full, dtype=np.float64).ravel()[1:]
        resid = np.concatenate(([r0], r_rest))
        grad = row0_vec.astype(np.float64) * r0 + col_norm.astype(np.float64) * (
            X_scaled[1:, :].T @ r_rest
        )
        viol = np.where(weights > 0, np.abs(grad), np.maximum(-grad, 0.0))
        raw = float(np.max(viol))
        # Cauchy-Schwarz denominator as in kkt_violation, rebuilt through the
        # identities above; round-off can make the expression dip below zero,
        # hence the clip. An exactly-fitting solution reports 0 like the stock
        # version's 0/0 guard.
        col_sq = np.maximum(
            col_norm.astype(np.float64) ** 2 - mu + row0_vec.astype(np.float64) ** 2,
            0.0,
        )
        scale = np.sqrt(col_sq) * np.linalg.norm(resid)
        if not np.any(scale > 0):
            return 0.0, raw
        scale = np.where(scale > 0, scale, np.inf)
        return float(np.max(viol / scale)), raw

    def solve_adelie_alm(self, problem):
        r"""Solve the NNLS problem with adelie BVLS + an augmented Lagrangian.

        Consumes an :class:`AdelieProblem` built by
        :meth:`construct_adelie_matrix_and_rhs` (or
        :meth:`_build_augmented_X` from a classic A, b pair) and returns
        ``(best_w, resid_full)``: the best weights and the plain residual
        ``A @ w - b`` aligned to A's rows (slot 0 = total-mass row), so the
        caller can compute chi2 without A.

        **Why an augmented Lagrangian.** Row 0 enforces
        :math:`\sum_j w_j = M_{tot}` with ``econ[0] = 1e-8``, so it contributes
        :math:`\tfrac{1}{2}\,10^{16}(\mathbf{1}^Tw-1)^2`. That is the
        :math:`\mu\to\infty` limit of a quadratic penalty, i.e. a hard equality
        constraint imposed by brute force. On omega Cen the row raises the
        condition number to 5e22. It also inflates adelie's convergence
        threshold: the stopping test is ``convg_measure <= tol * y_var`` with
        ``y_var = ||b||^2/n``, and ``b[0] = 1e8`` gives ``y_var = 1.04e12``, so
        the threshold becomes ~1e5 instead of ~1e-7. adelie then reports
        convergence after 2 iterations at ``sum(w) = 0``.

        ALM imposes the same constraint with a moderate ``mu`` by carrying a
        multiplier, so the matrix passed to adelie stays well scaled::

            row     = sqrt(mu) * 1',   target = sqrt(mu) * (1 + lam/mu)
            lam    <- lam - mu * (1'w - 1)

        A fixed penalty alone would leave a biased optimum. Updating the target
        through ``lam`` removes that bias without raising ``mu``: on omega Cen
        the constraint gap reaches 6e-11 with ``mu = 1e7``.

        Three implementation details are needed for this to be practical:

        1. ``mu`` is fixed. Raising it when convergence stalls restores the
           original scaling problem; one such attempt reached ``mu = 5.3e11``
           and stalled at a gap of 2e-4.
        2. With ``mu`` fixed the augmented matrix does not change between
           iterations (only the scalar target does), so it is built and
           column-scaled once.
        3. Each subproblem is warm-started from the previous adelie state.
           Consecutive subproblems differ by one number and converge in 1 to 2
           inner iterations.

        4. chi2 for the iterate is taken from ``state.resid`` rather than
           recomputed as ``A @ w - b``. The column scaling is exact, so
           ``X[1:] @ beta == A[1:] @ w`` and adelie's residual already holds
           everything except row 0 of A, which is one dot product over the
           orbits. Evaluating ``A @ w`` instead costs a full pass over the
           matrix (125 GiB for omega Cen) on every multiplier update.

        Items 2 and 3 reduced the per-iteration cost from 18.2 s to 0.6 s on
        NGC6278, which is what makes several hundred multiplier updates
        affordable. Item 4 matters on omega Cen rather than NGC6278: it is
        proportional to the size of the matrix, not to the solve.

        The inner solves are inexact, so the multiplier oscillates about its
        fixed point and the final iterate is arbitrary within that spread. The
        iterate with the lowest chi2 is returned instead.

        .. warning::
           **Known limitation: ``adelie_mu`` is an absolute constant but should
           be relative to the matrix scale.** The default 1e7 was validated on
           omega Cen and NGC6278, where the non-total-mass rows have column
           norms of order 1e4 to 1e5, so ``sqrt(mu) ~ 3e3`` is comparable to
           them. On a synthetic problem whose other rows are of order 1, the
           same mu leaves chi2 = 8.5 where 0 is attainable, while mu = 1e2
           reaches 1.6e-4. A galaxy whose constraint magnitudes differ from
           omega Cen's may therefore sit off the validated plateau, and the
           failure is silent. Until mu is set from the data, check the logged
           chi2 against a scipy solve on any new dataset.

        Parameters
        ----------
        problem : AdelieProblem

        Returns
        -------
        tuple (array, array)
            ``(best_w, resid_full)``: orbit weights and the plain residual
            aligned to A's rows.

        References
        ----------
        Hestenes, M. R. 1969, J. Optim. Theory Appl., 4, 303 -- augmented
            Lagrangian method (independently of Powell)
        Powell, M. J. D. 1969, in Optimization, ed. R. Fletcher (Academic
            Press), 283
        Nocedal, J. & Wright, S. J. 2006, Numerical Optimization, 2nd edn.
            (Springer), Chap. 17 -- standard treatment, including why the
            multiplier update removes the bias of a fixed finite penalty
        Yang, J. & Hastie, T. 2024a, arXiv:2410.03014 -- the BVLS coordinate
            descent solver called here
        Yang, J. & Hastie, T. 2024b, arXiv:2405.08631 -- the adelie package

        """
        n_threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))
        mu = self.adelie_mu
        sqrt_mu = np.sqrt(mu)
        X, col_norm, y = problem.X, problem.col_norm, problem.y
        dtype = X.dtype
        n_orbs = X.shape[1]

        lower = np.zeros(n_orbs, dtype=dtype)
        upper = np.full(n_orbs, np.inf, dtype=dtype)
        # adelie's bvls() infers a state dtype from X but defaults `weights`
        # to np.full(n, 1/n) (always float64); when X is float32 that
        # mismatch makes an internal `np.array(..., copy=False)` raise under
        # numpy>=2.0. Passing weights explicitly in X's dtype avoids it.
        weights_arr = np.full(X.shape[0], 1 / X.shape[0], dtype=dtype)
        # Hand adelie a matrix OBJECT rather than the raw ndarray, built once
        # outside the loop. Given an ndarray, bvls() recomputes
        #     X_vars = np.sum(weights[:, None] * X**2, axis=0)
        # on EVERY call, which materialises a full-size temporary - ~67 GiB per
        # ALM iterate at omega Cen in float32, times adelie_alm_iters. The
        # matrix object takes bvls()'s other branch, where the same quantity
        # comes from the compiled X.sq_mul() with no allocation at all.
        # matrix.dense wraps the existing buffer (verified: np.shares_memory),
        # so this costs nothing and the two X_vars agree to ~1e-15.
        X_solver = _adelie_matrix.dense(X, method="naive", n_threads=n_threads)

        lam = 0.0
        state = None
        best_chi2, best_w, best_it = np.inf, None, -1
        best_gap = np.nan
        # adelie returns an unconverged iterate WITHOUT raising when the inner
        # solve exhausts its budget, so the caller has to look. Tracked here
        # because a silently unconverged w becomes a silently wrong chi2 in
        # all_models.ecsv, indistinguishable from a real one.
        bvls_max_iters = self.adelie_bvls_max_iters
        inner_iters_max = 0
        n_saturated = 0
        for it in range(self.adelie_alm_iters):
            y[0] = sqrt_mu * (1.0 + lam / mu)
            state = _adelie_solver.bvls(
                X_solver,
                np.ascontiguousarray(y),
                lower,
                upper,
                weights=weights_arr,
                n_threads=n_threads,
                tol=self.adelie_tol,
                max_iters=bvls_max_iters,
                warm_start=state,
            )
            # state.iters counts coordinate descents for THIS call; the counter
            # restarts on every warm-started call, so compare it per call.
            inner_iters = int(state.iters)
            inner_iters_max = max(inner_iters_max, inner_iters)
            if inner_iters >= bvls_max_iters:
                n_saturated += 1
                if n_saturated == 1:  # once per solve, not once per iterate
                    self.logger.warning(
                        f"adelie ALM: BVLS reached max_iters={bvls_max_iters} at "
                        f"ALM iterate {it} - the inner solve did NOT converge and "
                        f"its weights are being used regardless. adelie_tol="
                        f"{self.adelie_tol:.1e} against {np.dtype(dtype).name} eps="
                        f"{np.finfo(dtype).eps:.1e}; a tolerance at or below eps "
                        "cannot be reached at this dtype."
                    )
            w = (np.asarray(state.beta).ravel() / col_norm).astype(np.float64)
            gap = float(w.sum() - 1.0)
            lam -= mu * gap
            # chi2 without touching A at all. X[1:] is A[1:] with unit-L2
            # column scaling and w = beta/col_norm, so X[1:] @ beta is exactly
            # A[1:] @ w, and adelie already returns resid = y - X @ beta from
            # the solve we just did. Only row 0 of A needs evaluating here,
            # because X replaces it with the ALM penalty row. This drops a full
            # pass over A (~125 GiB for omega Cen) from every iteration.
            # The astype is a no-op at the default nnls_dtype and a deliberate
            # 3 MB upcast at float32, where summing 371212 float32 terms would
            # lose precision that the old float64 accumulation had.
            resid = np.asarray(state.resid).ravel()[1:]
            resid = resid.astype(np.float64, copy=False)
            # row0_vec is bitwise A[0], so this equals the old float(A[0]@w)
            row0 = float(problem.row0_vec @ w) - problem.b0
            chi2 = row0 * row0 + float(resid @ resid)
            if chi2 < best_chi2:
                best_chi2, best_w, best_it = chi2, w.copy(), it
                best_gap = gap
            if abs(gap) < self.adelie_gap_tol:
                break

        self.logger.info(
            f"adelie ALM: {it + 1} iterations, mu={mu:.1e}, "
            f"final gap={gap:.2e}, best iterate {best_it}, "
            f"chi2={best_chi2:.4f}, sum(w)={best_w.sum():.10f}"
        )
        # The line above reports the FINAL gap, but the weights returned are
        # best_w, from iterate best_it. Those are different iterates whenever
        # the ALM is still oscillating, and best_w is selected on chi2 alone -
        # which measures ||Aw - b|| and says nothing about sum(w) = 1. So report
        # the feasibility of the iterate actually being returned, plus how hard
        # the inner solves had to work to produce it.
        self.logger.info(
            f"adelie ALM: returned iterate {best_it} has gap={best_gap:.2e} "
            f"(adelie_gap_tol={self.adelie_gap_tol:.1e}), "
            f"adelie_tol={self.adelie_tol:.1e} vs {np.dtype(dtype).name} eps="
            f"{np.finfo(dtype).eps:.1e}, peak BVLS iters in any "
            f"ALM iterate {inner_iters_max}/{bvls_max_iters}"
            + (
                f", {n_saturated} of {it + 1} ALM iterates UNCONVERGED"
                if n_saturated
                else ""
            )
        )
        # plain residual at the returned weights, aligned to A's rows, built
        # from X: rows 1.. via the exact column scaling (round-off-level
        # difference vs a gemv over A), row 0 via row0_vec. Serves both the
        # surrogate KKT below and solve()'s final chi2_vector.
        r0 = float(problem.row0_vec @ best_w) - problem.b0
        r_rest = y[1:].astype(np.float64) - X[1:, :] @ (
            col_norm.astype(np.float64) * best_w
        )
        resid_full = np.concatenate(([r0], r_rest))
        kkt, kkt_raw = self.kkt_violation_augmented(
            problem.row0_vec, problem.b0, X, col_norm, resid_full, best_w, mu
        )
        self.logger.info(
            f"adelie ALM: KKT violation scaled={kkt:.3e} (in [0,1]), raw={kkt_raw:.3e}"
        )
        # A fixed threshold cannot separate converged from unconverged across
        # datasets: omega Cen's good solution scores 5.7e-03 while a known
        # UNCONVERGED synthetic one scores 1.4e-05. What counts as small is
        # problem dependent. 0.1 therefore only catches unambiguous failures
        # (the stored omega Cen scipy solution scores 1.0, the maximum).
        # For a real convergence check, compare chi2 against a scipy solve.
        if kkt > 0.1:
            self.logger.warning(
                f"adelie ALM: scaled KKT violation {kkt:.3e} is close to the "
                "maximum of 1 - the solution is far from optimal. Check "
                "adelie_mu against a scipy solve on this dataset."
            )
        return best_w, resid_full

    def solve(self, orblib, ignore_existing_weights=False):
        """Solve for orbit weights

        **Note:** the returned chi2 values are not the same as
        ``LegacyWeightSolver.read_chi2`` - see the docstring for more info

        Apart from weight solving, the attributes ``orblib.intrinsic_masses``,
        ``orblib.projected_masses``, and ``orblib.vel_histograms`` are set
        via calling ``orblib.read_vel_histograms()``.

        Parameters
        ----------
        orblib : dyn.OrbitLibrary
        ignore_existing_weights : bool, optional
            If True, do not check for already existing weights and solve again.
            Default is False.

        Returns
        -------
        tuple
            (weights, chi2_all, chi2_kin, chi2_kinmap) where:
                -   weights : array, of orbit weights
                -   chi2_all : float, sum of squared residuals for intrinsic
                    masses, projected_masses and GH coefficients from h_1 to h_n
                -   chi2_kin : float sum of squared residuals for GH
                    coefficients h_1 to h_n
                -   chi2_kinmap : directly calculates the chi2 from the
                    kinematic maps

        """
        self.logger.info(f"Using WeightSolver: {__class__.__name__}/{self.nnls_solver}")
        if (not ignore_existing_weights) and self.weight_file_exists():
            results = ascii.read(self.weight_file, format="ecsv")
            self.logger.info("NNLS solution read from existing output")
            weights = results["weights"]
            chi2_tot = results.meta["chi2_tot"]
            chi2_kin = results.meta["chi2_kin"]
            chi2_kinmap = results.meta["chi2_kinmap"]
        else:
            # On the fused+streaming path the constructor drives per-set reads
            # itself (freeing each set after use), so no eager full read here.
            adelie_streaming = self.nnls_solver == "adelie" and self.stream_reads
            # Same deal for the blockwise Gram path: construct_gram_and_rhs_
            # blockwise drives its own per-set reads (freeing each set after
            # it is folded into the accumulator), so no eager full read here
            # either.
            gram_streaming = (
                self.gram_blockwise
                and self.nnls_solver in ("cvxopt", "admm")
                and self.stream_reads
            )
            # chi2_kinmap needs the library exactly as assembly received it.
            # Assembly restores the edge bins it zeroes, but streaming frees
            # each set's histograms and float32 rewrites hist.y in place.
            orblib_reusable = (
                not adelie_streaming
                and not gram_streaming
                and self.nnls_dtype != np.float32
            )
            if not adelie_streaming and not gram_streaming:
                orblib.read_vel_histograms()  # sets orblib.vel_histograms,
                # orblib.intrinsic_masses, and
                # orblib.projected_masses
                # downcast the retained orbit-library data before building the
                # NNLS matrix (no-op unless nnls_dtype is float32)
                _downcast_orblib(orblib, self.nnls_dtype)
            if self.nnls_solver == "adelie":
                # The fused constructor assembles adelie's augmented matrix X
                # directly from the orbit library - A is never built on this
                # path, which halves the resident set during the solve.
                if not _ADELIE_AVAILABLE:
                    text = (
                        "nnls_solver 'adelie' is not installed. Run: pip install adelie"
                    )
                    self.logger.error(text)
                    raise ImportError(text)
                try:
                    problem = self.construct_adelie_matrix_and_rhs(orblib)
                    weights, resid_full = self.solve_adelie_alm(problem)
                except Exception as e:
                    txt = (
                        f"Orblib {orblib.mod_dir}, ml={orblib.parset['ml']}"
                        f": adelie ALM solver error occured: {e} All weights "
                        "and chi2 set to nan. Consider trying scipy."
                    )
                    self.logger.warning(txt)
                    weights = np.full(orblib.n_orbs, np.nan)
            elif self.nnls_solver == "scipy":
                # scipy/cvxopt keep the classic A-based path. The adelie
                # branch above builds its augmented matrix directly instead.
                A, b = self.construct_nnls_matrix_and_rhs(orblib)
                # Normalize the data: building it on the adelie path would
                # hold a second full copy of A (~125 GiB for omega Cen).
                A_max = np.max(np.abs(A), axis=0)
                A_normalized = A / A_max
                b_max = np.max(np.abs(b))
                b_normalized = b / b_max
                try:
                    # Solve the NNLS problem with normalized data
                    x_normalized, rnorm = optimize.nnls(A_normalized, b_normalized)
                    # Scale the solution back to the original scale
                    weights = x_normalized * b_max / A_max

                except Exception as e:
                    txt = (
                        f"Orblib {orblib.mod_dir}, ml={orblib.parset['ml']}"
                        f": SciPy solver error occured: {e} All weights "
                        "and chi2 set to nan. Consider trying cvxopt."
                    )
                    self.logger.warning(txt)
                    weights = np.full(A.shape[1], np.nan)
            elif self.nnls_solver == "cvxopt":
                # Row 0 is the total-mass constraint: orbmat[0,:] is all ones
                # and econ[0] = max(|1 - total_mass|, 1e-8), which for a
                # normalised MGE lands on the 1e-8 FLOOR. So A[0,:] = 1e8 in
                # EVERY column - one constant row, six orders above the rest.
                # As a least-squares row it costs ~5 orders of magnitude of
                # conditioning, and P = An^T An squares that, putting the QP
                # past float64's limit (measured kappa(P) ~ 3e14, at which
                # cvxopt returns 'optimal' on a badly wrong answer).
                # cvxopt takes equality constraints natively, so state
                # sum(w) = total_mass exactly and drop the row.
                if self.gram_blockwise:
                    # P, q accumulated one row block at a time - A never
                    # exists. See construct_gram_and_rhs_blockwise's
                    # docstring for the ordering trap this must get right.
                    gram_problem = self.construct_gram_and_rhs_blockwise(orblib)
                    P, q = gram_problem.P, gram_problem.q
                    col_norm, b_max = gram_problem.col_norm, gram_problem.b_max
                else:
                    A, b = self.construct_nnls_matrix_and_rhs(orblib)
                    A_rest, b_rest = A[1:], b[1:]
                    # Unit-L2 column scaling (marginally better conditioned
                    # than the max-abs scaling the scipy path uses, and it
                    # matches the adelie path's _scale_columns). Null orbits
                    # give a zero norm; leave those columns unscaled rather
                    # than dividing by zero.
                    col_norm = np.linalg.norm(A_rest, axis=0)
                    col_norm[col_norm == 0] = 1.0
                    b_max = np.max(np.abs(b_rest))
                    if b_max == 0:
                        b_max = 1.0
                    A_normalized = A_rest / col_norm
                    b_normalized = b_rest / b_max
                    P = np.dot(A_normalized.T, A_normalized)
                    q = -1.0 * np.dot(A_normalized.T, b_normalized)
                # Vasiliev Eq. 7 diagonal ridge on the normalised P (no-op at
                # ridge_lambda=0). chi2 is still reported on the UNREGULARISED
                # problem - the raw-G quadratic form below never sees it.
                self._ridge_shift = _apply_diagonal_ridge(P, self.ridge_lambda)
                if self._ridge_shift > 0:
                    self.logger.info(
                        f"ridge: lambda={self.ridge_lambda:g}, absolute shift "
                        f"{self._ridge_shift:.6e} added to diag(P)"
                    )
                try:
                    # w = x * b_max / col_norm, so the mass constraint
                    # sum(w) = total_mass becomes (b_max/col_norm) . x = total_mass
                    solver = CvxoptNonNegSolver(
                        P,
                        q,
                        eq_coeff=b_max / col_norm,
                        eq_rhs=self.total_mass,
                        tol=self.cvxopt_tol,
                        maxiters=self.cvxopt_maxiters,
                        kktsolver=self.cvxopt_kktsolver,
                        show_progress=self.cvxopt_show_progress,
                        log_every=self.cvxopt_log_every,
                        logger=self.logger,
                    )
                    x_normalized = solver.beta
                    # Scale the solution back to the original scale
                    weights = x_normalized * b_max / col_norm
                    self.logger.info(
                        f"cvxopt QP: status={solver.status}, "
                        f"{solver.iterations} iterations, tol={self.cvxopt_tol:.1e}, "
                        f"gap={solver.gap}, elapsed={solver.elapsed:.2f}s, "
                        f"kktsolver={solver.kktsolver_used}, "
                        f"sum(w)={weights.sum():.10f} (target {self.total_mass:.10f})"
                    )
                    if not solver.success:
                        # 'unknown' means the iteration limit was reached or the
                        # search stalled; cvxopt still returns iterates, and
                        # without this they would become a chi2 like any other.
                        self.logger.warning(
                            f"cvxopt QP did NOT converge (status "
                            f"'{solver.status}' after {solver.iterations} "
                            f"iterations, maxiters={self.cvxopt_maxiters}). The "
                            "returned weights are the last iterate, not a "
                            "verified optimum."
                        )
                except Exception as e:
                    txt = (
                        f"Orblib {orblib.mod_dir}, ml={orblib.parset['ml']}"
                        f": CVXOPT solver error occured: {e} All weights "
                        "and chi2 set to nan. Consider trying scipy."
                    )
                    self.logger.warning(txt)
                    weights = np.full(orblib.n_orbs, np.nan)
            elif self.nnls_solver == "admm":
                # Same equality-drop scaling as the cvxopt path above (row 0
                # is the 1e8 total-mass row; see the comment there).
                free_p = self.admm_free_p and self.gram_blockwise
                gram_problem = None
                if self.gram_blockwise:
                    gram_problem = self.construct_gram_and_rhs_blockwise(orblib)
                    P, q = gram_problem.P, gram_problem.q
                    col_norm, b_max = gram_problem.col_norm, gram_problem.b_max
                else:
                    A, b = self.construct_nnls_matrix_and_rhs(orblib)
                    A_rest, b_rest = A[1:], b[1:]
                    col_norm = np.linalg.norm(A_rest, axis=0)
                    col_norm[col_norm == 0] = 1.0
                    b_max = np.max(np.abs(b_rest))
                    if b_max == 0:
                        b_max = 1.0
                    A_normalized = A_rest / col_norm
                    b_normalized = b_rest / b_max
                    P = np.dot(A_normalized.T, A_normalized)
                    q = -1.0 * np.dot(A_normalized.T, b_normalized)
                # same ridge as the cvxopt branch above; no-op at 0.0
                self._ridge_shift = _apply_diagonal_ridge(P, self.ridge_lambda)
                if self._ridge_shift > 0:
                    self.logger.info(
                        f"ridge: lambda={self.ridge_lambda:g}, absolute shift "
                        f"{self._ridge_shift:.6e} added to diag(P)"
                    )
                try:
                    solver = AdmmNonNegSolver(
                        P,
                        q,
                        eq_coeff=b_max / col_norm,
                        eq_rhs=self.total_mass,
                        rho=self.admm_rho,
                        max_iters=self.admm_max_iters,
                        tol=self.admm_tol,
                        logger=self.logger,
                        factor_in_place=free_p,
                    )
                    # With factor_in_place, C IS the old P buffer: every
                    # other reference to P must die NOW or nothing is freed.
                    if free_p:
                        # keep only what chi2 needs (all O(p) or tiny)
                        gp_chi2 = dict(
                            q=q,
                            b_sq_rest=gram_problem.b_sq_rest,
                            A_mass=gram_problem.A_mass,
                            b_mass=gram_problem.b_mass,
                            row0_vec=gram_problem.row0_vec,
                            b0=gram_problem.b0,
                        )
                        del gram_problem, P
                    x_normalized = solver.beta
                    weights = x_normalized * b_max / col_norm
                    self.logger.info(
                        f"ADMM QP: status={solver.status}, "
                        f"{solver.iterations} iterations, rho={solver.rho:.6e}, "
                        f"pri_resid={solver.r_pri:.3e}, dual_resid={solver.r_dual:.3e}, "
                        f"elapsed={solver.elapsed:.2f}s, "
                        f"sum(w)={weights.sum():.10f} (target {self.total_mass:.10f})"
                    )
                    if not solver.success:
                        self.logger.warning(
                            f"ADMM QP did NOT converge (status '{solver.status}' "
                            f"after {solver.iterations} iterations, "
                            f"max_iters={self.admm_max_iters}, tol={self.admm_tol:.1e}). "
                            "The returned weights are the last iterate, not a "
                            "verified optimum."
                        )
                except Exception as e:
                    txt = (
                        f"Orblib {orblib.mod_dir}, ml={orblib.parset['ml']}"
                        f": ADMM solver error occured: {e} All weights "
                        "and chi2 set to nan. Consider trying cvxopt or scipy."
                    )
                    self.logger.warning(txt)
                    weights = np.full(orblib.n_orbs, np.nan)
            else:
                text = "Unknown nnls_solver"
                self.logger.error(text)
                raise ValueError(text)
            if not np.isnan(weights[0]):
                # calculate chi2s
                if self.nnls_solver == "adelie":
                    # chi2 without A: the plain residual at the returned
                    # weights, aligned to A's rows, plus the total-mass row
                    # term. Algebraically identical to (A@w - b)**2; differs
                    # only in gemv rounding (quantified in the perf notes).
                    row0_resid = float(problem.row0_vec @ weights) - problem.b0
                    chi2_vector = chi2_vector_from_residuals(
                        resid_full, row0_resid * row0_resid
                    )
                    chi2_tot = np.sum(chi2_vector)
                    chi2_kin = np.sum(chi2_vector[self.n_mass_constraints :])
                    # X is dead from here on, and chi2_kinmap below may read a
                    # fresh orbit library (~165 GiB at production scale). Drop
                    # X first so the two peaks do not stack - glibc unmaps
                    # numpy's buffers at `del`, measured at 100% reclaim.
                    del problem, resid_full
                elif self.nnls_solver == "admm" and free_p:
                    # chi2 from C alone: P was CONSUMED by the in-place
                    # factorization. Quadratic form via the Cholesky identity
                    # w'Gw - 2w'v == b_max^2 * (x'Px + 2x'q), with x'Px from
                    # solver.gram_quadratic_form (no production path pre-ridges
                    # P today, so extra_shift=0; pass lam*mean_diag(P) if that
                    # ever changes). Splitting chi2_mass via A_mass as in the
                    # raw-G branch below. Equivalence with that branch is
                    # algebraic, not approximate; pinned in
                    # dev_tests/test_admm_free_p.py.
                    row0_resid = float(gp_chi2["row0_vec"] @ weights) - gp_chi2["b0"]
                    # extra_shift = the ridge the caller pre-added to P, so
                    # the identity recovers the UNREGULARISED quadratic form
                    g_pp = solver.gram_quadratic_form(
                        x_normalized, extra_shift=self._ridge_shift
                    )
                    chi2_rest = (
                        b_max**2 * (g_pp + 2.0 * float(x_normalized @ gp_chi2["q"]))
                        + gp_chi2["b_sq_rest"]
                    )
                    mass_resid = gp_chi2["A_mass"] @ weights - gp_chi2["b_mass"]
                    chi2_mass = float(np.dot(mass_resid, mass_resid))
                    chi2_kin = chi2_rest - chi2_mass
                    chi2_tot = row0_resid * row0_resid + chi2_rest
                    del gp_chi2, mass_resid
                elif self.nnls_solver in ("cvxopt", "admm") and self.gram_blockwise:
                    # chi2 from the Gram form, no pass over A: chi2_rest =
                    # w'Gw - 2 w'v + ||b_rest||^2 (mass + kinematic rows
                    # together, raw scale - NOT the column-normalized P/q),
                    # split into chi2_mass/chi2_kin via the small
                    # materialized A_mass block, since chi2_kin (unlike
                    # chi2_tot) excludes the mass rows. See
                    # construct_gram_and_rhs_blockwise and
                    # dev_tests/test_gram_blockwise.py::test_chi2_matches_residual.
                    row0_resid = (
                        float(gram_problem.row0_vec @ weights) - gram_problem.b0
                    )
                    Gw = gram_problem.G @ weights
                    chi2_rest = (
                        float(weights @ Gw)
                        - 2.0 * float(weights @ gram_problem.v)
                        + gram_problem.b_sq_rest
                    )
                    mass_resid = gram_problem.A_mass @ weights - gram_problem.b_mass
                    chi2_mass = float(np.dot(mass_resid, mass_resid))
                    chi2_kin = chi2_rest - chi2_mass
                    chi2_tot = row0_resid * row0_resid + chi2_rest
                    del gram_problem, Gw, mass_resid
                else:
                    chi2_vector = (np.dot(A, weights) - b) ** 2.0
                    chi2_tot = np.sum(chi2_vector)
                    chi2_kin = np.sum(chi2_vector[self.n_mass_constraints :])
                chi2_kinmap = self.chi2_kinmap(
                    weights, orblib=orblib if orblib_reusable else None
                )
                # save the output
                results = table.Table()
                results["weights"] = weights
                # add chi2 to meta data
                results.meta = {
                    "chi2_tot": chi2_tot,
                    "chi2_kin": chi2_kin,
                    "chi2_kinmap": chi2_kinmap,
                }
                results.write(self.weight_file, format="ascii.ecsv", overwrite=True)
                self.logger.info("NNLS problem solved and chi2 calculated.")
            else:
                chi2_tot = chi2_kin = chi2_kinmap = np.nan
            # delete existing .yaml files and copy current config file
            # into model directory
            self.config.copy_config_file(self.direc_with_ml)
        return weights, chi2_tot, chi2_kin, chi2_kinmap


def _make_cvxopt_kktsolver(P_np, a_np, logger=None, log_every=1):
    """Build a custom KKT solver for cvxopt's coneqp, routed through scipy's
    LAPACK instead of cvxopt's own vendored (single-threaded) OpenBLAS.

    cvxopt's generic KKT path does not exploit the structure of our problem:
    G = -I (diagonal) and A is a SINGLE dense row (the total-mass equality).
    Eliminating the z-block gives K = P + W^-2, and the equality row
    eliminates by a scalar Schur complement, i.e. ONE Cholesky per interior
    point iteration plus one triangular solve for K^-1 a^T. cvxopt's own
    generic path costs ~12-15 dpotrf-equivalents per iteration instead of
    one, on top of running through BLAS that is stuck at 0.06 Tflop/s at any
    thread count (measured; scipy's dpotrf/dpotrs hit 0.93-1.33 Tflop/s at
    p=12000-16000). Net measured speedup: 1.76x (p=2000) to 5.95x
    (p=12000), with iterates identical to the default path to ~1e-15.

    SIGN CONVENTION (got this wrong twice - preserve the comment): cvxopt
    solves for (ux, uy, uz) satisfying
        G.ux + uz_scaled = bz    with   G = -I,
    and the f() callback must OVERWRITE bz with W.uz on return (not uz
    itself). With G = -I this makes K = P + W^-2, the reduced right-hand
    side is rx = bx - d*bz (d = di**2, di = W["di"]), and the returned
    bz must be -di*(ux + z). Getting either sign wrong produces
    "ValueError: domain error" plus overflow warnings deep in coneqp, not a
    clean traceback at the call site.

    MEMORY: one extra p x p float64 array is allocated per factorization
    (a copy of P with d added to the diagonal, since dpotrf overwrites its
    input and P must survive to the next iteration) - 16.2 GB at omega
    Cen's p=45000. This is on top of P itself, which coneqp already holds.

    Parameters
    ----------
    P_np : ndarray (p, p)
        Quadratic part of the objective, as a plain numpy array (not a
        cvxopt.matrix). Not modified.
    a_np : ndarray (p,)
        The single equality row's coefficients.
    logger : logging.Logger, optional
        If given, a heartbeat is logged at INFO every `log_every`
        factorizations - the only way to distinguish a slow solve from a
        hung one, since cvxopt's own progress printout goes to stdout (and
        is off by default here).
    log_every : int
        Heartbeat cadence in factorizations. 1 logs every factorization.

    Returns
    -------
    callable
        A function F(W) -> f(bx, by, bz), matching cvxopt's kktsolver
        protocol (see cvxopt.solvers.coneqp docs, "Providing a custom KKT
        solver").
    """
    p = P_np.shape[0]
    a_col = a_np.reshape(p, 1)
    state = {"n": 0, "t0": time.time()}

    def F(W):
        di = np.asarray(W["di"]).ravel()
        d = di * di
        K = np.array(P_np, order="F", copy=True)  # the ONE extra p x p
        K.flat[:: p + 1] += d
        C, info = dpotrf(K, lower=1, clean=0, overwrite_a=1)
        if info:
            raise ArithmeticError(f"P + W^-2 not PD (info={info})")
        Ka = dpotrs(C, a_col, lower=1)[0].ravel()
        s = float(a_np @ Ka)
        state["n"] += 1
        if logger is not None and log_every > 0 and state["n"] % log_every == 0:
            logger.info(
                f"cvxopt custom kktsolver: factorization {state['n']}, "
                f"{time.time() - state['t0']:.1f}s elapsed"
            )

        def f(bx, by, bz):
            x = np.array(bx).ravel()
            y = float(by[0])
            z = np.array(bz).ravel()
            Kr = dpotrs(C, (x - d * z).reshape(p, 1), lower=1)[0].ravel()
            uy = (float(a_np @ Kr) - y) / s
            ux = Kr - uy * Ka
            bx[:] = cvxopt.matrix(ux)
            by[:] = cvxopt.matrix([uy])
            bz[:] = cvxopt.matrix(-di * (ux + z))

        return f

    return F


class CvxoptNonNegSolver:
    """Solver for NNLS problem using CVXOPT

    Solves the QP problem:
        argmin (1/2 beta^T P beta + q beta T)
        subject to (component-wise) beta > 0
        and, optionally, a single linear equality eq_coeff . beta = eq_rhs

    By default (``kktsolver="custom"``) this routes the interior-point KKT
    reduction through scipy's LAPACK rather than cvxopt's own vendored
    OpenBLAS, which is built single-threaded (measured 0.06 Tflop/s at any
    thread count, vs scipy's 0.93-1.33 Tflop/s). At omega Cen's p=45000 the
    default path burned 4262s of wall time in ONE Cholesky factorization in
    a production run (py-spy: stuck in cvxopt/misc.py:factor while 46 other
    threads idled); the custom path finishes the equivalent solve in
    minutes. See ``_make_cvxopt_kktsolver`` for the algorithm and the sign
    convention that took two attempts to get right. The custom solver is
    only valid for exactly one equality row; with zero or more than one it
    is not used, and why is logged at INFO.

    A multi-hour cvxopt solve prints NOTHING to stdout by default
    (``show_progress=False``, preserved here) - the same failure class as a
    previous incident where a silent 31.5h run finished 0/90 models. To
    make that impossible: the status/iteration count/gap/wall time are
    always logged at INFO after the solve regardless of ``show_progress``,
    the custom kktsolver logs a heartbeat every ``log_every``
    factorizations, and a non-``optimal`` status is logged at WARNING
    rather than silently treated as a converged answer.

    Parameters
    ----------
    P : array (p, p)
        quadratic part of objective function
    q : array (p,)
        linear part of objective function
    eq_coeff : array (p,), optional
        coefficients of a single linear equality constraint
    eq_rhs : float, optional
        right-hand side of the equality constraint
    tol : float
        shared value for cvxopt's abstol/reltol/feastol
    maxiters : int
        cvxopt's interior-point iteration cap
    kktsolver : str
        "custom" (default) to use the scipy-LAPACK KKT reduction described
        above when there is exactly one equality row, "default" to always
        use cvxopt's own KKT solver.
    show_progress : bool
        forwarded to cvxopt.solvers.options["show_progress"]; controls
        cvxopt's own stdout printout only. Default False, matching prior
        behaviour. Independent of the INFO-level logging described above,
        which always happens.
    log_every : int
        heartbeat cadence (in factorizations) for the custom kktsolver.
        Only has an effect when the custom kktsolver is actually used.
    logger : logging.Logger, optional
        logger used for the heartbeat and the WARNING on non-convergence.
        If None, a module-level logger is used.

    Attributes
    ----------
    success : bool
        whether solver was successful (status == "optimal")
    status : str
        cvxopt's reported status ("optimal", "unknown", ...)
    iterations : int
        number of interior-point iterations
    gap : float
        final duality gap reported by cvxopt
    elapsed : float
        wall time of the cvxopt.solvers.qp call, in seconds
    kktsolver_used : str
        "custom" or "default", whichever was actually used (may differ
        from the requested ``kktsolver`` if the fallback triggered)
    beta : array (p,)
        solution

    """

    def __init__(
        self,
        P=None,
        q=None,
        eq_coeff=None,
        eq_rhs=None,
        tol=1e-9,
        maxiters=200,
        kktsolver="custom",
        show_progress=False,
        log_every=1,
        logger=None,
    ):
        p = P.shape[0]
        log = logger or logging.getLogger(f"{__name__}.CvxoptNonNegSolver")
        # -I as a SPARSE matrix. The dense np.identity(p) this replaces is
        # p**2 doubles - 16.2 GiB at omega Cen's 45000 orbits - to express a
        # constraint with p nonzeros, and it costs per-iteration work too.
        G = cvxopt.spmatrix(-1.0, range(p), range(p))
        h = cvxopt.matrix(np.zeros(p))
        kwargs = {}
        n_eq_rows = 0
        if eq_coeff is not None:
            # A single linear equality, e.g. total mass. cvxopt handles this
            # natively; expressing it instead as a heavily weighted ROW of the
            # least-squares matrix is what destroys the conditioning of P.
            eq_coeff_arr = np.asarray(eq_coeff, dtype=float)
            n_eq_rows = 1 if eq_coeff_arr.ndim == 1 else eq_coeff_arr.shape[0]
            kwargs["A"] = cvxopt.matrix(eq_coeff_arr.reshape(n_eq_rows, p))
            kwargs["b"] = cvxopt.matrix(np.atleast_1d(np.asarray(eq_rhs, dtype=float)))
        self.kktsolver_used = "default"
        if kktsolver == "custom":
            if n_eq_rows == 1:
                kwargs["kktsolver"] = _make_cvxopt_kktsolver(
                    np.asarray(P, dtype=float),
                    np.asarray(eq_coeff, dtype=float).reshape(p),
                    logger=log,
                    log_every=log_every,
                )
                self.kktsolver_used = "custom"
            else:
                log.info(
                    "cvxopt_kktsolver='custom' requested but there "
                    f"{'is no equality row' if n_eq_rows == 0 else f'are {n_eq_rows} equality rows'} "
                    "(the custom KKT reduction is only valid for exactly "
                    "one); falling back to cvxopt's default KKT solver."
                )
        elif kktsolver != "default":
            raise ValueError(f"Unknown kktsolver {kktsolver!r}")
        # cvxopt.solvers.options is module-global state; leave it as found.
        saved = dict(cvxopt.solvers.options)
        cvxopt.solvers.options.update(
            {
                "abstol": tol,
                "reltol": tol,
                "feastol": tol,
                "maxiters": int(maxiters),
                "show_progress": bool(show_progress),
            }
        )
        t0 = time.time()
        try:
            sol = cvxopt.solvers.qp(cvxopt.matrix(P), cvxopt.matrix(q), G, h, **kwargs)
        finally:
            cvxopt.solvers.options.clear()
            cvxopt.solvers.options.update(saved)
        self.elapsed = time.time() - t0
        self.status = sol["status"]
        self.success = self.status == "optimal"
        self.iterations = sol.get("iterations")
        self.gap = sol.get("gap")
        self.beta = np.squeeze(np.array(sol["x"]))
        log.info(
            f"cvxopt coneqp done: status={self.status}, "
            f"iterations={self.iterations}, gap={self.gap}, "
            f"elapsed={self.elapsed:.2f}s, kktsolver={self.kktsolver_used}"
        )
        if not self.success:
            log.warning(
                f"cvxopt coneqp returned non-optimal status "
                f"'{self.status}' after {self.iterations} iterations; "
                "the returned iterate is NOT a verified optimum."
            )


class AdmmNonNegSolver:
    """Solver for the same NNLS QP as CvxoptNonNegSolver, using ADMM with a
    FIXED penalty rho instead of cvxopt's interior point method.

    Solves:
        argmin 0.5 beta^T P beta + q^T beta
        subject to beta >= 0  and  eq_coeff . beta = eq_rhs   (optional)

    via the splitting (w = z, non-negativity on z, equality kept on w)::

        w-update:  min .5 w'Pw + q'w + (rho/2)||w - z + u||^2  s.t. a'w = M
        z-update:  z = max(w + u, 0)
        u-update:  u += w - z

    The w-update is the same single-equality KKT system as
    ``_make_cvxopt_kktsolver``, but here the matrix being factored,
    M = P + rho*I, does NOT depend on the iterate (rho is fixed) - so it is
    factored ONCE, up front, and every iteration is two triangular solves
    against that one Cholesky factor. This is the entire point of ADMM over
    interior point for this problem: interior point must refactor
    P + W^-2 every iteration because W changes.

    Measured at p=45000 (synthetic P): factor 24.1s at 1.263 Tflop/s, 400
    iterations in 499s, total 8.72 min, peak RSS 15.6 GB (vs ~422 GB for
    the production cvxopt path at the same p). Final residuals reached
    pri=2.4e-10, dual=5.1e-08.

    KNOWN BEHAVIOURS (documented, not bugs):

    - rho changes the ITERATION COUNT by up to 3 orders of magnitude but not
      the converged answer - verified identical to 1e-15 across 4 decades
      of rho. It is a pure cost knob, not a modelling choice. The default
      (``rho=None``) picks ``trace(P) / p``, which matches P's scale.
    - ADMM produces EXACT ZEROS (from the z-update's clip to 0), so its
      support is genuinely sparse. cvxopt's interior point does NOT do
      this - measured 1000/1000 nonzero weights at almost every problem
      scale. Neither is a defect: they are different optima of the same
      convex problem seen through different stopping rules, and ADMM's
      sparsity is a legitimate property of the fixed-point iterate, not
      thresholding applied after the fact.
    - A diagonal ridge added to P (P.flat[::p+1] += lam*scale) both
      improves conditioning and collapses the ADMM iteration count
      sharply (measured 3335 -> 30 iterations at lam=10, p=8000). This
      solver does not add a ridge itself, but a caller doing so gets a
      compounding benefit, not just a conditioning one.

    Parameters
    ----------
    P : array (p, p)
        quadratic part of the objective
    q : array (p,)
        linear part of the objective
    eq_coeff : array (p,), optional
        coefficients of a single linear equality constraint (e.g. total
        mass). Required in practice for this problem's use in dynamite,
        but the algebra tolerates it being all-ones etc.
    eq_rhs : float, optional
        right-hand side of the equality constraint
    rho : float, optional
        fixed ADMM penalty. If None, defaults to trace(P)/p (see above).
    max_iters : int
        iteration cap
    tol : float
        convergence tolerance on the normalised primal and dual residuals
        (see ``r_pri``/``r_dual`` below); both must be below ``tol``.
    logger : logging.Logger, optional
        used only if the caller wants heartbeats in the future; currently
        unused internally beyond being accepted (kept symmetric with
        CvxoptNonNegSolver's signature).

    Attributes
    ----------
    success : bool
        whether both residuals were below tol within max_iters
    status : str
        "optimal" or "unknown" (mirrors cvxopt's vocabulary so callers can
        treat the two solvers uniformly)
    iterations : int
        number of ADMM iterations run
    rho : float
        the rho actually used (resolved from None if applicable)
    r_pri, r_dual : float
        final normalised primal/dual residuals
    elapsed : float
        wall time of factorization + iteration, in seconds
    beta : array (p,)
        solution (the z-iterate, which carries the exact zeros)

    """

    def __init__(
        self,
        P=None,
        q=None,
        eq_coeff=None,
        eq_rhs=None,
        rho=None,
        max_iters=4000,
        tol=1e-11,
        logger=None,
        factor_in_place=False,
    ):
        p = P.shape[0]
        a_np = np.asarray(eq_coeff, dtype=float).reshape(p)
        m = float(eq_rhs)
        q_np = np.asarray(q, dtype=float).reshape(p)
        P_np = np.asarray(P, dtype=float)

        if rho is None:
            rho = float(np.trace(P_np)) / p
        self.rho = float(rho)

        t0 = time.time()
        if factor_in_place:
            # DESTROY P: add rho to its diagonal and factor it into its own
            # buffer. The caller must drop every reference to P afterwards -
            # the Cholesky factor C *is* the old P's memory. Saves one full
            # p x p copy (~16 GB at omega Cen) and leaves only C resident
            # for the iterations.
            if not (P_np.flags["F_CONTIGUOUS"] or P_np.flags["C_CONTIGUOUS"]):
                raise ValueError("factor_in_place requires a contiguous P")
            if P_np.dtype != np.float64:
                raise ValueError("factor_in_place requires a float64 P")
        else:
            P_np = np.array(P_np, order="F", copy=True)  # the ONE extra p x p
        P_np.flat[:: p + 1] += self.rho
        C, info = dpotrf(P_np, lower=1, clean=0, overwrite_a=1)
        del P_np
        if info:
            raise ArithmeticError(f"P + rho*I not PD (info={info})")
        # dpotrf(clean=0) leaves the UNUSED triangle of C as garbage: nothing
        # may ever read C with a dense product. dpotrs below and dtrmv in
        # gram_quadratic_form both respect triangularity.
        self.chol_factor = C
        Ka = dpotrs(C, a_np.reshape(p, 1), lower=1)[0].ravel()
        s = float(a_np @ Ka)

        w = np.zeros(p)
        z = np.zeros(p)
        u = np.zeros(p)
        r_pri = r_dual = np.inf
        it = 0
        for it in range(1, int(max_iters) + 1):
            r = dpotrs(C, (self.rho * (z - u) - q_np).reshape(p, 1), lower=1)[0].ravel()
            nu = (float(a_np @ r) - m) / s
            w = r - nu * Ka
            z_old = z
            z = np.maximum(w + u, 0.0)
            u = u + w - z
            r_pri = float(np.linalg.norm(w - z))
            r_dual = float(self.rho * np.linalg.norm(z - z_old))
            scale = max(np.linalg.norm(w), np.linalg.norm(z), 1.0)
            r_pri /= scale
            r_dual /= scale
            if r_pri < tol and r_dual < tol:
                break
        self.elapsed = time.time() - t0
        self.iterations = it
        self.r_pri = r_pri
        self.r_dual = r_dual
        self.success = bool(r_pri < tol and r_dual < tol)
        self.status = "optimal" if self.success else "unknown"
        self.beta = z
        log = logger or logging.getLogger(f"{__name__}.AdmmNonNegSolver")
        log.info(
            f"ADMM done: status={self.status}, iterations={self.iterations}, "
            f"rho={self.rho:.6e}, r_pri={self.r_pri:.3e}, "
            f"r_dual={self.r_dual:.3e}, elapsed={self.elapsed:.2f}s"
        )
        if not self.success:
            log.warning(
                f"ADMM did not converge to tol={tol:.1e} within "
                f"{self.iterations} iterations (r_pri={self.r_pri:.3e}, "
                f"r_dual={self.r_dual:.3e}); the returned iterate is NOT a "
                "verified optimum."
            )

    def gram_quadratic_form(self, w, extra_shift=0.0):
        """``w' P_solver w`` from the Cholesky factor alone - no P needed.

        C factors M = P_solver + rho*I, so

            w'P_solver w = ||C' w||^2 - (rho + extra_shift) * ||w||^2

        where ``extra_shift`` is any diagonal the CALLER added to P before
        handing it over (e.g. a pre-added ridge lam*mean_diag(P)): the
        identity subtracts exactly what was factored relative to the
        reference form wanted. Production chi2 lives on the UNREGULARISED
        Gram matrix, so a caller that pre-ridged P passes
        extra_shift=lam*mean_diag.

        Accuracy: pure cancellation rounding, measured rel err == eps*R with
        R = shift*||w||^2 / w'Pw (PM_grid/_diag_21_admm_free/NOTES.md):
        ~1e-16 on converged solutions, worst measured 7.5e-12 on an exact
        smallest-eigenvector direction. Uses dtrmv because dpotrf(clean=0)
        leaves C's unused triangle as garbage - a dense C.T@w is WRONG by
        O(1) (pinned in dev_tests/test_admm_free_p.py).
        """
        C = self.chol_factor
        Lt_w = dtrmv(C, np.asarray(w, dtype=float).ravel(), lower=1, trans=1)
        shift = self.rho + float(extra_shift)
        return float(Lt_w @ Lt_w - shift * float(w @ w))


# end
