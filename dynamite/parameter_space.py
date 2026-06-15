import sys
import copy
import logging
import itertools
import numpy as np
from astropy.table import Table
from dynamite import parameter_space as parspace

# ---------------------------------------------------------------------------
# Bayesian Optimization: training-data extraction pipeline
# (used by BayesOptGenerator)
# ---------------------------------------------------------------------------

def extract_gp_training_data(all_models_table, parspace, which_chi2='kinchi2'):
    """Extract normalized GP training data from an AllModels.table.

    Parameter columns hold *par_value* (physical units). Converts to
    *raw_value* (log10 for logarithmic params) and normalizes each free
    parameter to [0, 1] using par_generator_settings lo/hi (raw space).

    Returns
    -------
    X_norm : np.ndarray (n_valid, n_free) in [0, 1]
    y : np.ndarray (n_valid,)  -- chi2 values
    free_param_names : list[str]
    lo_raw, hi_raw : np.ndarray (n_free,)  -- bounds in raw_value space
    """
    t = all_models_table
    done_mask = np.asarray(t['all_done'], dtype=bool)
    chi2_col = np.asarray(t[which_chi2], dtype=float)
    finite_mask = np.isfinite(chi2_col)
    valid_mask = done_mask & finite_mask

    valid_rows = t[valid_mask]
    y = chi2_col[valid_mask]

    free_params = [p for p in parspace if not p.fixed]
    free_param_names = [p.name for p in free_params]
    n_free = len(free_params)
    n_valid = int(np.sum(valid_mask))

    lo_raw = np.array([p.par_generator_settings['lo'] for p in free_params],
                      dtype=float)
    hi_raw = np.array([p.par_generator_settings['hi'] for p in free_params],
                      dtype=float)

    raw_matrix = np.empty((n_valid, n_free), dtype=float)
    for j, par in enumerate(free_params):
        par_values = np.asarray(valid_rows[par.name], dtype=float)
        if par.logarithmic:
            raw_matrix[:, j] = np.log10(par_values)
        else:
            raw_matrix[:, j] = par_values

    span = hi_raw - lo_raw
    X_norm = (raw_matrix - lo_raw) / span
    return X_norm, y, free_param_names, lo_raw, hi_raw


def denormalize_to_raw(X_norm, lo_raw, hi_raw):
    """Convert normalized [0,1] inputs back to raw_value space."""
    X_norm = np.atleast_2d(X_norm)
    span = hi_raw - lo_raw
    return X_norm * span + lo_raw


def raw_to_par_values(raw_values_row, free_params):
    """Convert a single row of raw_values to par_values (physical units)."""
    raw = np.asarray(raw_values_row, dtype=float)
    par_vals = np.empty_like(raw)
    for j, par in enumerate(free_params):
        if par.logarithmic:
            par_vals[j] = 10.0 ** raw[j]
        else:
            par_vals[j] = raw[j]
    return par_vals


def get_qobs_from_system(system):
    """Return qobs of the TriaxialVisibleComponent, or None if absent.

    Traverses system.cmp_list defensively. Returns None if:
    - system is None or has no cmp_list
    - no TriaxialVisibleComponent (exact class, not BarDiskComponent subclass)
    - qobs is nan or non-finite
    """
    if system is None or not hasattr(system, 'cmp_list'):
        return None
    for cmp in system.cmp_list:
        if cmp.__class__.__name__ == 'TriaxialVisibleComponent':
            qobs = getattr(cmp, 'qobs', None)
            if qobs is not None and np.isfinite(qobs):
                return float(qobs)
    return None


class Parameter(object):
    """Parameter of a model

    Parameters
    ----------
    name : string
        the parameter name (specific components have specific parameter names)
    fixed : Bool
        whether or not to fix this parameter during parameter searches
    LaTeX : string
        a ```LaTeX`` format string to use in plots
    sformat : string
        a format string for printing parameter values
    value : float
        the value of this parameter in a model; the config file contains an
        initial value, this is updated during the parameter search; this value
        can be in log or linear units, depending on the config file
    par_generator_settings : dict
        settings for the parameter generator
    logarithmic : Bool
        whether or not this parameter is specified in log units

    """
    attributes = []
    def __init__(self,
                 name=None,
                 fixed=False,
                 LaTeX=None,
                 sformat=None,
                 value=None,
                 par_generator_settings=None,
                 logarithmic=False,
                 ):
        self.logger = logging.getLogger(f'{__name__}.{__class__.__name__}')
        self.name = name
        self.fixed = fixed
        self.LaTeX = LaTeX
        self.sformat = sformat
        self.raw_value = value
        self.par_generator_settings = par_generator_settings
        self.logarithmic = logarithmic
        self.__class__.attributes = list(self.__dict__.keys())

    def update(self, **kwargs):
        """update the parameter
        """
        for k, v in kwargs.items():
            if k not in self.__class__.attributes:
                text = (f'Invalid parameter key {k}. Allowed keys: '
                        f'{str(tuple(self.__class__.attributes))}')
                self.logger.error(text)
                raise ValueError(text)
            setattr(self, k, v)

    def validate(self):
        """validate the parameter
        """
        if sorted(self.__class__.attributes) != sorted(self.__dict__.keys()):
            text = (f'Parameter attributes can only be '
                    f'{str(tuple(self.__class__.attributes))}, '
                    f'not {str(tuple(self.__dict__.keys()))}')
            self.logger.error(text)
            raise ValueError(text)

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.__dict__})')

    @property
    def par_value(self):
        """ getter method for par_value to be used like an attribute
        """
        return self.get_par_value_from_raw_value(self.raw_value)

    @par_value.setter
    def par_value(self, new_par_value):
        """ setter method for par_value to be used like an attribute
        """
        self.raw_value = self.get_raw_value_from_par_value(new_par_value)

    def get_par_value_from_raw_value(self, raw_value):
        """Get parameter value from the raw value

        In `raw` values, linearly-sized steps are taken during parameter
        searches. Currently there is only one possible `raw` transformation,
        going to log units. Future ones may include, e.g. isotropic
        transformations of viewing angles.

        Parameters
        ----------
        raw_value : float
            the raw parameter value

        Returns
        -------
        float
            the parameter value

        """
        if self.logarithmic is True:
            par_value = 10.**raw_value
        else:
            par_value = raw_value
        return par_value

    def get_raw_value_from_par_value(self, par_value):
        """Get raw parameter value from parameter

        In `raw` values, linearly-sized steps are taken during parameter
        searches. Currently there is only one possible `raw` transformation,
        going to log units. Future ones may include, e.g. isotropic
        transformations of viewing angles.

        Parameters
        ----------
        par_value : float
            the parameter value

        Returns
        -------
        float
            the raw parameter value

        """
        if self.logarithmic is True:
            raw_value = np.log10(par_value)
        else:
            raw_value = par_value
        return raw_value


class ParameterSpace(list):
    """A list of all ``Parameter`` objects  in the ``Model``

    Parameters
    ----------
    system : a ``dyn.physical_system.System`` object

    """
    def __init__(self, system):
        self.logger = logging.getLogger(f'{__name__}.{__class__.__name__}')
        for cmp in system.cmp_list:
            for par in cmp.parameters:
                self.append(par)
                self.system = system
        for par in system.parameters:
            self.append(par)

        self.par_names = []
        for par in self:
            self.par_names.append(par.name)

        self.n_par = len(self)
        self.n_par_fixed = len([p for p in self if p.fixed])
        self.n_par_free = self.n_par - self.n_par_fixed

    def __repr__(self):
        return (f'{self.__class__.__name__}({[p for p in self]}, '
                f'{self.__dict__})')

    def get_param_value_from_raw_value(self, raw_value):
        """Get parameter values from raw parameters

        In `raw` values, linearly-sized steps are taken during parameter
        searches. Currently there is only one possible `raw` transformation,
        going to log units. Future ones may include, e.g. isotropic
        transformations of viewing angles.

        Parameters
        ----------
        raw_value : list of floats
            list of raw parameter values for all models

        Returns
        -------
        list of floats
            list of parameter value

        """
        par_val = [p.get_par_value_from_raw_value(rv0)
                   for (rv0, p) in zip(raw_value, self)]
        return par_val

    def get_raw_value_from_param_value(self, par_val):
        """Get raw parameter values from parameters

        In `raw` values, linearly-sized steps are taken during parameter
        searches. Currently there is only one possible `raw` transformation,
        going to log units. Future ones may include, e.g. isotropic
        transformations of viewing angles.

        Parameters
        ----------
        par_val : list of floats
            list of parameter values for all models

        Returns
        -------
        list of floats
            list of raw parameter value

        """
        raw_value = [p.get_raw_value_from_par_value(pv0)
                     for (pv0, p) in zip(par_val, self)]
        return raw_value

    def get_parameter_from_name(self, name):
        """Get a ``Parameter`` from the name

        Parameters
        ----------
        name : string

        Returns
        -------
        ``dyn.parameter_space.Parameter``

        """
        name_array = np.array(self.par_names)
        idx = np.where(name_array == name)
        self.logger.debug(f'Checking unique parameter name {name}...')
        error_msg = f"There should be 1 and only 1 parameter named {name}"
        assert len(idx[0]) == 1, error_msg
        self.logger.debug('...check ok.')
        parameter = self[idx[0][0]]
        return parameter

    def get_parset(self):
        """
        Get parset as row of an Astropy Table

        Returns
        -------
        parset : row of an Astropy Table
            Contains the values of the individual parameters.

        """
        t = Table()
        for par in self:
            t[par.name] = [par.par_value]
        # extract 0th - i.e. the only - row from the table
        parset = t[0]
        return parset

    def validate_parset(self, parset):
        """
        Validates a parameter set

        Validate the values of each component's parameters by calling the
        individual components' validate_parameter methods. Does the same
        for system parameters. Used by the parameter generators.

        Parameters
        ----------
        parset : list of Parameter objects

        Returns
        -------
        Bool
            True if validation was successful, False otherwise

        """
        isvalid = True
        for comp in self.system.cmp_list:
            par = {comp.get_parname(p.name):p.raw_value for p in parset \
                   if p.name.rfind(f'{comp.name}')>=0}
            isvalid = isvalid and comp.validate_parset(par)
        par = {p.name:p.raw_value for p in parset \
               if p.name in [n.name for n in self.system.parameters]}
        isvalid = isvalid and self.system.validate_parset(par)
        return isvalid

    def validate_parspace(self):
        """
        Validates a parameter set

        Validate the values of each component's parameters by calling the
        individual components' validate_parameter methods and check whether
        parameters are within the specified lo/hi bounds.
        Does the same for system parameters.

        Raises
        ------
        ValueError
            If checks fail due to various reasons.

        Returns
        -------
        None.

        """
        for comp in self.system.cmp_list:
            par = {comp.get_parname(p.name):p.raw_value for p in self \
                   if p.name.rfind(f'{comp.name}')>=0}
            if not comp.validate_parset(par):
                text = f'Parameters {par} of component {comp.name} failed ' \
                       'to validate.'
                self.logger.error(text)
                raise ValueError(text)
        par = {p.name:p.raw_value for p in self \
               if p.name in [n.name for n in self.system.parameters]}
        if not self.system.validate_parset(par):
            text = f'System parameters {par} failed to validate.'
            self.logger.error(text)
            raise ValueError(text)
        # Now, check for violoating allowed parameter ranges
        for p in self:
            if type(p.par_generator_settings) is dict:
                try:
                    lo = p.par_generator_settings['lo']
                except:
                    text = f"Parameter {p.name}={p.raw_value}: cannot check " \
                           "lower bound due to missing 'lo' setting."
                    self.logger.debug(text)
                else:
                    if lo > p.raw_value:
                        text = f'Parameter {p.name}={p.raw_value} out of ' \
                               f'bounds: violates {lo}<={p.raw_value}.'
                        self.logger.error(text)
                        raise ValueError(text)
                try:
                    hi = p.par_generator_settings['hi']
                except:
                    text = f"Parameter {p.name}={p.raw_value}: cannot check " \
                           "upper bound due to missing 'hi' setting."
                    self.logger.debug(text)
                else:
                    if p.raw_value > hi:
                        text = f'Parameter {p.name}={p.raw_value} out of ' \
                               f'bounds: violates {p.raw_value}<={hi}.'
                        self.logger.error(text)
                        raise ValueError(text)
            else:
                self.logger.debug(f"Parameter {p.name}={p.raw_value}: cannot "\
                    "check bounds due to missing 'lo' and 'hi' settings.")


class ParameterGenerator(object):
    """Abstract class for ``ParameterGenerator``

    ``ParameterGenerator`` have methods to generate new sets of parameters to
    evaluate models based on existing models. This is an abstrct class, specific
    implementations should be implemented as child-classes (e.g.
    ``LegacyGridSearch``). Every implementation may have a method
    ``check_specific_stopping_criteria`` and must have a method
    ``specific_generate_method`` which
    define the stopping criteria and parameter generation algorithm for that
    implementation. These are exectuted in addition to ``generic`` methods,
    which are defined in this parent ``ParameterGenerator`` class.

    Parameters
    ----------
    par_space : ``dyn.parameter_space.ParameterSpace`` object
    parspace_settings : dict
        parameter space settings
    name : string
        the name of the particular ParameterGenerator sub-class

    """
    def __init__(self,
                 par_space=[],
                 parspace_settings=None,
                 name=None):
        self.logger = logging.getLogger(f'{__name__}.{__class__.__name__}')
        self.par_space = par_space
        if not parspace_settings:
            text = 'ParameterGenerator needs parspace_settings'
            self.logger.error(text)
            raise ValueError(text)
        self.parspace_settings = parspace_settings
        self.chi2 = self.parspace_settings.get('which_chi2')
        self.status = {}
        self.name = name
        self.lo = []
        self.hi = []
        try:
            for par in self.par_space:
                settings = par.par_generator_settings
                if par.fixed is False:
                    self.lo.append(settings['lo'])
                    self.hi.append(settings['hi'])
                else:
                    self.lo.append(None)
                    self.hi.append(None)
        except:
            text = 'ParameterGenerator: non-fixed parameters ' + \
                   'need hi and lo settings'
            self.logger.error(text)
            raise ValueError(text)
        try:
            stop_crit = parspace_settings['stopping_criteria']
        except:
            text = 'ParameterGenerator: need stopping criteria'
            self.logger.error(text)
            raise ValueError(text)
        if not stop_crit.get('n_max_mods') and \
           not stop_crit.get('n_max_iter'):
            text = 'ParameterGenerator: need n_max_mods and ' + \
                   'n_max_iter stopping criteria settings'
            self.logger.error(text)
            raise ValueError(text)

    def generate(self,
                 current_models=None,
                 kw_specific_generate_method={}):
        """Generate new parameter sets.

        This is a wrapper method around the ``specific_generate_method`` of
        child generator classes. This wrapper does the following:

            1.   evaluates stopping criteria, and stops if necessary
            2.   runs the ``specific_generate_method`` of the child class, which
                 updates ``self.model_list`` with a list of proposal models
            3.   removes previously run and invalid models from ``self.model_list``
            4.   converts parameters from raw_values to par_values
            5.   adds new, valid models to ``current_models.table``
            6.   update and return the status dictionary

        Parameters
        ----------
        current_models : dynamite.AllModels
        kw_specific_generate_method : dict, optional
            keyword arguments passed to the specific_generate_method of the
            child class. The default is {}

        Returns
        -------
        dict
            a status dictionary, with entries:
                - stop: bool, whether or not any stopping criteria are met
                - n_new_models: int
            and additional Bool entries for the indivdidual stopping criteria:
                - last_iter_added_no_new_models
                - n_max_mods_reached
                - n_max_iter_reached
                - plus any criteria specific to the child class

        """
        if current_models is None:
            errormsg = "current_models needs to be a valid " \
                       "dynamite.AllModels instance"
            self.logger.error(errormsg)
            raise ValueError(errormsg)
        self.current_models = current_models
        self.check_stopping_criteria()
        if len(self.current_models.table)==0:
            this_iter = 0
        else:
            this_iter = np.max(self.current_models.table['which_iter']) + 1
        # check whether we need to do anything in the first place...
        newmodels = 0
        if not self.status['stop']:
            self.specific_generate_method(**kw_specific_generate_method)
            # Add new models to current_models.table
            for m in self.model_list:
                if self._is_newmodel(m, eps=1e-10):
                    self.add_model(m, n_iter=this_iter)
                    newmodels += 1
        else:
            self.model_list = []
        self.logger.info(f'{self.name} added {newmodels} new model(s) out of '
                         f'{len(self.model_list)}')
        # combine first two iterations by calling the generator again...
        if this_iter==0 and newmodels>0:
            newmodels0 = newmodels
            this_iter += 1
            self.specific_generate_method(**kw_specific_generate_method)
            # Add new models to current_models.table
            for m in self.model_list:
                if self._is_newmodel(m, eps=1e-10):
                    self.add_model(m, n_iter=this_iter)
                    newmodels += 1
            self.logger.info(f'{self.name} added {newmodels-newmodels0} new '
                             f'model(s) out of {len(self.model_list)}')
        self.status['n_new_models'] = newmodels
        self.status['last_iter_added_no_new_models'] = newmodels==0
        self.status['stop'] = newmodels==0
        return self.status

    def add_model(self, model=None, n_iter=0):
        """
        Add a model

        Adds the model (a list of ``Parameter`` objects) to the table
        ``self.current_models``. The datetime64 column is populated with the
        current timestamp numpy.datetime64('now', 'ms'). The 'which_iter' column
        is populated with the argument value n_iter.

        **Note** - here, `model` refers to a list of Parameter objects, not a
        ``Model`` object. TODO: clarify the naming confusion.

        Parameters
        ----------
        model : List of Parameter objects
        n_iter : integer
            value to write in 'which_iter' column, optional. The default is 0.

        Raises
        ------
        ValueError
            If no or empty model is given.

        Returns
        -------
        None.

        """
        if not model:
            self.logger.error('No or empty model')
            raise ValueError('No or empty model')
        row = [p.par_value for p in model]
        # for all columns after parameters, add an entry to this row
        idx_start = self.par_space.n_par
        idx_end = len(self.current_models.table.colnames)
        for i in range(idx_start, idx_end):
            if self.current_models.table.columns[i].name == 'time_modified':
                # current time
                val = str(np.datetime64('now', 's'))
            elif self.current_models.table.columns[i].name == 'which_iter':
                # iteration
                val = n_iter
            elif self.current_models.table.columns[i].name == 'directory':
                val = ''
            else:
                # empty/nan/'None' entry for all other columns
                val = self.current_models.table.columns[i].dtype.type(None)
            row.append(val)
        self.current_models.table.add_row(row)

    def check_stopping_criteria(self):
        """Check stopping criteria

        This is a wrapper which checks both the generic stopping criteria and
        also the ``specific_stopping_criteria`` revelant for any particular
        ``ParameterGenerator`` used.

        Returns
        -------
        None
            Sets the attribute ``self.status['stop']`` to True if any of the
            stopping criteria are met, else to False.

        """
        self.status['stop'] = False
        if len(self.current_models.table) > 0:
            # never stop when current_models is empty
            self.check_generic_stopping_criteria()
            self.check_specific_stopping_criteria()
            if any(v for v in self.status.values() if type(v) is bool):
                self.status['stop'] = True
                self.logger.info(f'Stopping criteria met: {self.status}.')

    def check_generic_stopping_criteria(self):
        """check generic stopping criteria

        Returns
        -------
        None
            Sets the attributes ``self.status['n_max_mods_reached']`` and
            ``self.status['n_max_iter_reached']``.

        """
        self.status['n_max_mods_reached'] = \
            len(self.current_models.table) \
                >= self.parspace_settings['stopping_criteria']['n_max_mods']
        self.status['n_max_iter_reached'] = \
            np.max(self.current_models.table['which_iter']) \
                >= self.parspace_settings['stopping_criteria']['n_max_iter']
        # iii) ...

    def check_specific_stopping_criteria(self):
        """checks specific stopping criteria

        If the last iteration did not improve the chi2 by at least
        min_delta_chi2, then stop. May be overwritten or extended in
        each ``ParameterGenerator`` class.

        Returns
        -------
        None
            Sets the attribute ``self.status['min_delta_chi2_reached']``

        """
        # stop if...
        # (i) if iter>1, last iteration did not improve chi2 by min_delta_chi2
        self.status['min_delta_chi2_reached'] = False
        last_iter = np.max(self.current_models.table['which_iter'])
        if last_iter > 0:
            last_chi2 = np.nan
            while np.isnan(last_chi2): # look for non-nan (kin)chi2 value
                if last_iter <= 0:
                    return
                mask = self.current_models.table['which_iter'] == last_iter
                models0 = self.current_models.table[mask]
                last_chi2 = np.nanmin(models0[self.chi2])
                last_iter -= 1
            if last_iter < 0:
                return
            mask = self.current_models.table['which_iter'] <= last_iter
            models1 = self.current_models.table[mask]
            if len(models1) == 0:
                return
            previous_chi2 = np.nanmin(models1[self.chi2])
            if np.isnan(previous_chi2):
                return
            # Don't use abs() so we stop on increasing chi2 values, too:
            delta_chi2 = previous_chi2 - last_chi2
            if self.min_delta_chi2_rel is not None:
                if delta_chi2 / previous_chi2 < self.min_delta_chi2_rel:
                    self.status['min_delta_chi2_reached'] = True
            else:
                if delta_chi2 < self.min_delta_chi2_abs:
                    self.status['min_delta_chi2_reached'] = True
        # (ii) if step_size < min_step_size for all params
        #       => dealt with by grid_walk (doesn't create such models)

    def _is_newmodel(self, model, eps=1e-6):
        """
        Check if model is new

        Checks whether model has valid parameter values and it is a new model
        (i.e., its parameter set does not exist in self.current_models).

        Parameters
        ----------
        model : A self.model_list element (list of Parameter objects),
                mandatory
        eps : Used for numerical comparison (relative difference w.r.t.
              model values), default is 1e-6

        Returns
        -------
        isnew : True if model is a new model, False otherwise.

        """
        if any(map(lambda t: not isinstance(t, parspace.Parameter), model)):
            self.logger.error('Model arg. must be list of Parameter objects')
            raise ValueError('Model arg. must be list of Parameter objects')
        if not self.par_space.validate_parset(model):
            isnew = False
        else:
            isnew = True
            model_values = [p.par_value for p in model]
            if len(self.current_models.table) > 0:
                for mod in self.current_models.table[self.par_space.par_names]:
                    if np.allclose(list(mod), model_values, rtol=eps):
                        isnew = False
                        break
        return isnew

    def clip(self, value, mini, maxi):
        """
        clip to lo/hi

        Clips value to the interval [mini, maxi]. Similar to the numpy.clip()
        method. If mini==maxi, that value is returned.

        Parameters
        ----------
        value : numeric value
        mini : numeric value
        maxi : numeric value

        Raises
        ------
        ValueError if mini > maxi

        Returns
        -------
        min(max(mini, value), maxi)

        """
        logger = logging.getLogger(f'{__name__}.{__class__.__name__}')
        if mini <= maxi:
            return np.clip(value, mini, maxi)
        else:
            text = 'Clip error: minimum must be less than or equal to maximum'
            logger.error(text)
            raise ValueError(text)

    def specific_generate_method(self):
        """
        Placeholder.

        This is a placeholder. Specific ``ParameterGenerator`` sub-classes
        should have their own ``specific_generate_method`` methods.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
            set ``self.model_list`` to be the list of newly generated models

        """
        return


class LegacyGridSearch(ParameterGenerator):
    """Search around all reasonable models

    This is the method used by previous code versions AKA schwpy. See docstrings
    for ``specific_generate_method`` and ``check_specific_stopping_criteria``
    for full description.

    Parameters
    ----------
    par_space : ``dyn.parameter_space.ParameterSpace`` object
    parspace_settings : dict

    """
    def __init__(self, par_space=[], parspace_settings=None):
        super().__init__(par_space=par_space,
                         parspace_settings=parspace_settings,
                         name='LegacyGridSearch')
        self.logger = logging.getLogger(f'{__name__}.{__class__.__name__}')
        # We need a local parameter copy because we don't want to change the
        # minstep in the original par_space:
        self.new_parset = [copy.deepcopy(p) for p in self.par_space]
        self.step = []
        self.minstep = []
        try:
            for par in self.par_space:
                settings = par.par_generator_settings
                if par.fixed is False:
                    self.step.append(settings['step'])
                    # Use 'minstep' value if present, otherwise use 'step'.
                    # Explicitly set minstep=0 to allow arbitrarily
                    # small steps, not recommended.
                    self.minstep.append(settings['minstep'] \
                        if 'minstep' in settings else settings['step'])
                else:
                    self.step.append(None)
                    self.minstep.append(None)
        except:
            text = 'LegacyGridSearch: non-fixed parameters need step setting'
            self.logger.error(text)
            raise ValueError(text)
        try:
            self.thresh = \
            self.parspace_settings['generator_settings']['threshold_del_chi2']
        except:
            text = 'LegacyGridSearch: need generator_settings - ' + \
                'threshold_del_chi2 (absolute or scaled - see documentation)'
            self.logger.error(text)
            raise ValueError(text)
        stop_crit = parspace_settings['stopping_criteria']
        stop_abs = 'min_delta_chi2_abs' in stop_crit
        stop_rel = 'min_delta_chi2_rel' in stop_crit
        if (stop_abs and stop_rel) or not (stop_abs or stop_rel):
            text = 'LegacyGridSearch: specify exactly one of the ' + \
                   'options min_delta_chi2_abs, min_delta_chi2_rel'
            self.logger.error(text)
            raise ValueError(text)
        if stop_abs:
            self.min_delta_chi2_abs = stop_crit['min_delta_chi2_abs']
        else:
            self.min_delta_chi2_abs = None
        if stop_rel:
            self.min_delta_chi2_rel = stop_crit['min_delta_chi2_rel']
        else:
            self.min_delta_chi2_rel = None

    def specific_generate_method(self, **kwargs):
        r"""
        Generates new models

        Starts at the initial point. Start the iteration: (i) find all models
        with :math:`|\chi^2 - \chi_\mathrm{min}^2|` within the specified
        threshold, (ii) for each model within the threshold, seed new models by
        independently take a step :math:`\pm 1` of size ``step``. If no new
        models are seeded at the end of an iteration, then divide all parameter
        stepsizes by two till their specified ``minstep`` are reached.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
            sets ``self.model_list`` is the list of new models.

        """
        if len(self.current_models.table) == 0:
            # The 'zeroth iteration' results in only one model
            # (all parameters at their .raw_value level)
            self.model_list = [[p for p in self.par_space]]
            return ###########################################################
        if len(self.current_models.table) == 1: # 'first' iteration
            prop_mask = [True]
        else:
            min_chi2 = np.nanmin(self.current_models.table[self.chi2])
            if np.isnan(min_chi2):
                text = 'All (kin)chi2 values are nan.'
                self.logger.error(text)
                raise ValueError(text)
            prop_mask = \
                abs(self.current_models.table[self.chi2]-min_chi2)<=self.thresh
        prop_list = self.current_models.table[prop_mask]
        self.model_list = []
        step_ok = True
        while step_ok and len(self.model_list) == 0:
            for paridx, par in enumerate(self.new_parset):
                if par.fixed: # parameter fixed -> do nothing
                    continue
                lo = self.lo[paridx] #par.par_generator_settings['lo']
                hi = self.hi[paridx] #par.par_generator_settings['hi']
                step = self.step[paridx] #par.par_generator_settings['step']
                minstep = self.minstep[paridx]
                for m in prop_list: # for all models within threshold_del_chi2
                    for p in self.new_parset:
                        p.par_value = m[p.name]
                    raw_center = self.new_parset[paridx].raw_value
                    for s in [-1, 1]:
                        new_raw_value = np.clip(raw_center + s*step, lo, hi)
                        if abs(new_raw_value-par.raw_value) \
                           >= minstep - sys.float_info.epsilon:
                            self.new_parset[paridx].raw_value = new_raw_value
                            if self._is_newmodel(self.new_parset, eps=1e-10):
                                self.model_list.append\
                                  ([copy.deepcopy(p) for p in self.new_parset])
#                                    (copy.deepcopy(self.new_parset))
            # If no new models: cut stepsize in half & try again
            if len(self.model_list) == 0:
                step_ok = False
                for par in [p for p in self.new_parset if not p.fixed]:
                    paridx = self.new_parset.index(par)
                    minstep = self.minstep[paridx]
                    if self.step[paridx]/2 >= minstep:
                        self.step[paridx] /= 2
                        # the following line is just to record the step size
                        # in self.new_parset and can be commented out...
                        par.par_generator_settings['step'] = self.step[paridx]
                        step_ok = True
        return


class GridWalk(ParameterGenerator):
    """Walk after the current best fit

    See docstrings for ``specific_generate_method`` and
    ``check_specific_stopping_criteria`` for full description.

    Parameters
    ----------
    par_space : ``dyn.parameter_space.ParameterSpace`` object
    parspace_settings : dict

    """
    def __init__(self,
                 par_space=[],
                 parspace_settings=None):
        super().__init__(par_space=par_space,
                         parspace_settings=parspace_settings,
                         name='GridWalk')
        self.logger = logging.getLogger(f'{__name__}.{__class__.__name__}')
        self.step = []
        self.minstep = []
        try:
            for par in self.par_space:
                settings = par.par_generator_settings
                if par.fixed is False:
                    self.step.append(settings['step'])
                    # use 'minstep' value if present, otherwise use 'step'
                    self.minstep.append(settings['minstep'] \
                        if 'minstep' in settings else settings['step'])
                else:
                    self.step.append(None)
                    self.minstep.append(None)
        except:
            text = 'GridWalk: non-fixed parameters need step setting'
            self.logger.error(text)
            raise ValueError(text)
        stop_crit = parspace_settings['stopping_criteria']
        stop_abs = 'min_delta_chi2_abs' in stop_crit
        stop_rel = 'min_delta_chi2_rel' in stop_crit
        if (stop_abs and stop_rel) or not (stop_abs or stop_rel):
            text = 'GridWalk: specify exactly one of the ' + \
                   'options min_delta_chi2_abs, min_delta_chi2_rel'
            self.logger.error(text)
            raise ValueError(text)
        if stop_abs:
            self.min_delta_chi2_abs = stop_crit['min_delta_chi2_abs']
        else:
            self.min_delta_chi2_abs = None
        if stop_rel:
            self.min_delta_chi2_rel = stop_crit['min_delta_chi2_rel']
        else:
            self.min_delta_chi2_rel = None

    def specific_generate_method(self, **kwargs):
        """
        Generates new models

        The center of the grid walk is the parameter set with the smallest chi2
        value, depending on the parameter space setting 'which_chi2'.

        Parameters
        ----------
        None.

        Raises
        ------
        None.

        Returns
        -------
        None.
            sets the list ``self.model_list`` of new models.
        """
        if len(self.current_models.table) == 0:
            # The 'zeroth iteration' results in only one model
            # (all parameters at their .raw_value level)
            self.model_list = [[p for p in self.par_space]]
        else: # Subsequent iterations...
            if len(self.current_models.table) == 1: # 'first' iteration
                center_idx = 0
            else:
                # center criterion: min(chi2)
                center_idx = np.nanargmin(self.current_models.table[self.chi2])
            n_par = self.par_space.n_par
            center = list(self.current_models.table[center_idx])[:n_par]
            raw_center = self.par_space.get_raw_value_from_param_value(center)
            self.logger.debug(f'center: {center}')
            # Build model_list by walking the grid
            self.model_list = []
            self.grid_walk(center=raw_center)
            # for m in self.model_list:
            #     self.logger.debug(f'{[(p.name, p.raw_value) for p in m]}')

    def grid_walk(self, center=None, par=None, eps=1e-6):
        """
        Walks the grid

        Walks the grid defined by ``self.par_space.par_generator_settings``
        attributes. Clips parameter values to lo/hi attributes. If clipping
        violates the minstep attribute, the resulting model(s) will not be
        created. If the minstep attribute is missing, the step attribute will be
        used instead. Explicitly set minstep=0 to allow arbitrarily small steps
        (not recommended).

        Parameters
        ----------
        center : List of center coordinates. Must be in the same sequence as
                 the parameters in self.par_space. Mandatory argument.
        par : Internal use only. Gives the parameter to start with. Set
              automatically in the recursive process. The default is None.
        eps : Used for numerical comparison (relative tolerance), default 1e-6

        Raises
        ------
        ValueError if center is not specified or fixed parameters != center.

        Returns
        -------
        None. Sets self.model_list to the resulting models.

        """
        if center is None:
            text = 'Need center'
            self.logger.error(text)
            raise ValueError(text)
        if not par:
            par = self.par_space[0]
        paridx = self.par_space.index(par)
        self.logger.debug(f'Call with paridx={paridx}, '
                          f'n_par={self.par_space.n_par}')

        if par.fixed:
            raw_values = [par.raw_value]
            if abs(center[paridx] - par.raw_value) > eps:
                text='Something is wrong: fixed parameter value not in center'
                self.logger.error(text)
                raise ValueError(text)
        else:
            lo = self.lo[paridx]
            hi = self.hi[paridx]
            step = self.step[paridx]
            minstep = self.minstep[paridx]
            # up to 3 *distinct* raw_values (clipped lo, mid, hi values)
            raw_values = []
            # start with lo...
            delta = center[paridx] - self.clip(center[paridx] - step, lo, hi)
            if abs(delta) >= minstep - sys.float_info.epsilon:
                raw_values.append(self.clip(center[paridx] - step, lo, hi))
            # now mid... tol(erance) is necessary in case minstep < eps
            if len(raw_values) > 0:
                # check for raw values differing by more than eps...
                tol = abs(self.clip(center[paridx], lo, hi) - raw_values[0])
                if abs(raw_values[0]) > eps: # relative tolerance usable=?
                    tol /= abs(raw_values[0])
                if tol > eps:
                    raw_values.append(self.clip(center[paridx], lo, hi))
            else:
                raw_values.append(self.clip(center[paridx], lo, hi))
            # and now hi...
            delta = self.clip(center[paridx] + step, lo, hi) - center[paridx]
            if abs(delta) >= minstep - sys.float_info.epsilon:
                tol = abs(self.clip(center[paridx]+step,lo,hi)-raw_values[-1])
                if abs(raw_values[-1]) > eps:
                    tol /= abs(raw_values[-1])
                if tol > eps:
                    raw_values.append(self.clip(center[paridx]+step, lo, hi))

        for raw_value in raw_values:
            parcpy = copy.deepcopy(par)
            parcpy.raw_value = raw_value
            if not self.model_list: # add first entry if model_list is empty
                self.model_list = [[parcpy]]
                models_prev = [[]]
                self.logger.debug('new model list, starting w/parameter '
                                  f'{parcpy.name}')
            elif parcpy.name in [p.name for p in self.model_list[0]]:
                # in this case, create new (partial) model by copying last
                # models and setting the new parameter raw_value
                for m in models_prev:
                    new_model = m + [parcpy]
                    self.model_list.append(new_model)
                self.logger.debug(f'{parcpy.name} is in '
                      f'{[p.name for p in self.model_list[0]]}, '
                      f'added {parcpy.name}={parcpy.raw_value}')
            else: # new parameter: append it to existing (partial) models
                models_prev = copy.deepcopy(self.model_list)
                for m in self.model_list:
                    m.append(parcpy)
                self.logger.debug(f'new parameter {parcpy.name}='
                                  f'{parcpy.raw_value}')

        # call recursively until all paramaters are done:
        if paridx < self.par_space.n_par - 1:
            self.grid_walk(center=center, par=self.par_space[paridx+1])


class BayesOptGenerator(ParameterGenerator):
    """Bayesian Optimization parameter generator (BoTorch backend).

    Fits a Gaussian Process surrogate to chi2(parameters) in normalized
    raw_value space and proposes a batch of candidates per iteration by
    maximizing qLogExpectedImprovement. The triaxiality constraint on
    (q, p, u) shape parameters is enforced via BoTorch nonlinear
    inequality constraints when a TriaxialVisibleComponent is present.

    generator_settings keys (with defaults):
        batch_size : int = 8
        n_orblib_configs : int = 4
        n_ml_per_config : int = 2
        n_initial_random : int = 10
        acquisition_type : str = 'qLogEI'
        max_gp_variance_threshold : float = 1.0
        min_ei_threshold : float = -1.5

    stopping_criteria: exactly one of min_delta_chi2_abs / min_delta_chi2_rel
    (same rule as GridWalk).
    """

    def __init__(self, par_space=[], parspace_settings=None):
        super().__init__(par_space=par_space,
                         parspace_settings=parspace_settings,
                         name='BayesOptGenerator')
        self.logger = logging.getLogger(
            f'{__name__}.{self.__class__.__name__}')

        gen = (parspace_settings or {}).get('generator_settings') or {}
        self.batch_size = gen.get('batch_size', 8)
        self.n_orblib_configs = gen.get('n_orblib_configs', 4)
        self.n_ml_per_config = gen.get('n_ml_per_config', 2)
        self.n_initial_random = gen.get('n_initial_random', 10)
        self.acquisition_type = gen.get('acquisition_type', 'qLogEI')
        self.max_gp_variance_threshold = \
            gen.get('max_gp_variance_threshold', 1.0)
        self.min_ei_threshold = gen.get('min_ei_threshold', -1.5)

        # Exactly one of the chi2-delta backstops (GridWalk pattern).
        stop_crit = (parspace_settings or {}).get('stopping_criteria') or {}
        stop_abs = 'min_delta_chi2_abs' in stop_crit
        stop_rel = 'min_delta_chi2_rel' in stop_crit
        if (stop_abs and stop_rel) or not (stop_abs or stop_rel):
            text = ('BayesOptGenerator: specify exactly one of '
                    'min_delta_chi2_abs, min_delta_chi2_rel '
                    'in stopping_criteria')
            self.logger.error(text)
            raise ValueError(text)
        self.min_delta_chi2_abs = stop_crit.get('min_delta_chi2_abs')
        self.min_delta_chi2_rel = stop_crit.get('min_delta_chi2_rel')

        # Free-parameter bookkeeping (indices into the full par_space).
        self.free_par_idx = [i for i, p in enumerate(self.par_space)
                             if not p.fixed]
        self.free_params = [self.par_space[i] for i in self.free_par_idx]
        self.lo_free = [self.lo[i] for i in self.free_par_idx]
        self.hi_free = [self.hi[i] for i in self.free_par_idx]
        self.free_param_names = [p.name for p in self.free_params]

        # Position of q / p / u among the FREE parameters (for triaxiality).
        # Parameter names carry a component suffix (e.g. 'q-stars'), so strip
        # everything after the first '-' before comparing the base name.
        self._free_qpu_idx = {}
        for axis in ('q', 'p', 'u'):
            for j, p in enumerate(self.free_params):
                if p.name.split('-')[0] == axis:
                    self._free_qpu_idx[axis] = j
                    break

        # qobs from the system that owns the parameter space (if any).
        system = getattr(self.par_space, 'system', None)
        self.qobs = get_qobs_from_system(system)

        # GP state (set lazily during specific_generate_method).
        self._gp_model = None
        self._last_acq_value = None

    # --- candidate <-> Parameter conversion ----------------------------

    def _raw_free_to_model(self, raw_free):
        """Build one model (list of Parameter, all params) from a raw_free row.

        Fixed parameters keep their canonical raw_value; free parameters take
        the proposed raw values (in free-parameter order).
        """
        new_parset = [copy.deepcopy(p) for p in self.par_space]
        for j, full_idx in enumerate(self.free_par_idx):
            new_parset[full_idx].raw_value = float(raw_free[j])
        return new_parset

    def _raw_free_matrix_to_model_list(self, raw_free_matrix):
        """Convert a matrix of raw free-parameter values to a model list."""
        return [self._raw_free_to_model(row) for row in raw_free_matrix]

    def _norm_bounds_arrays(self):
        """Return (lo_raw, hi_raw) numpy arrays over free parameters."""
        return (np.array(self.lo_free, dtype=float),
                np.array(self.hi_free, dtype=float))

    def _sobol_unit(self, n):
        """Return n Sobol points in the unit cube [0,1]^n_free (numpy)."""
        import torch
        from torch.quasirandom import SobolEngine
        eng = SobolEngine(dimension=len(self.free_par_idx), scramble=True)
        return eng.draw(n).to(dtype=torch.double).numpy()

    def _project_unit_to_feasible_qpu(self, X_unit):
        """Project unit-cube samples so (q,p,u) satisfy the triaxiality box.

        Operates in normalized [0,1] space. Converts q/p/u columns to raw
        values, enforces p>=q and clips u to its feasibility window, then
        converts back to normalized. No-op if qobs is None or q/p/u not free.
        """
        if self.qobs is None or not all(k in self._free_qpu_idx
                                        for k in ('q', 'p', 'u')):
            return X_unit
        lo_raw, hi_raw = self._norm_bounds_arrays()
        span = hi_raw - lo_raw
        raw = X_unit * span + lo_raw          # (n, n_free) raw values
        jq = self._free_qpu_idx['q']
        jp = self._free_qpu_idx['p']
        ju = self._free_qpu_idx['u']
        q = raw[:, jq]
        p = raw[:, jp]
        p = np.maximum(p, q)                  # enforce p >= q
        u_lo = np.maximum(q / self.qobs, p)
        u_hi = np.minimum(p / self.qobs, 1.0)
        mid = 0.5 * (u_lo + u_hi)
        good = u_hi > u_lo
        u = raw[:, ju].copy()
        u[good] = np.clip(u[good], u_lo[good], u_hi[good])
        u[~good] = mid[~good]
        raw[:, jp] = p
        raw[:, ju] = u
        X_proj = (raw - lo_raw) / span
        return np.clip(X_proj, 0.0, 1.0)

    def _propose_random_batch(self):
        """Sobol random proposals (warm-up), structured for orblib reuse."""
        n_orblib = max(1, self.batch_size // max(1, self.n_ml_per_config))
        lo_raw, hi_raw = self._norm_bounds_arrays()
        span = hi_raw - lo_raw

        base_unit = self._sobol_unit(n_orblib)
        base_unit = self._project_unit_to_feasible_qpu(base_unit)

        ml_free_j = None
        for j, p in enumerate(self.free_params):
            if p.name == 'ml':
                ml_free_j = j
                break

        rows_unit = []
        for k in range(n_orblib):
            if ml_free_j is None or self.n_ml_per_config <= 1:
                # No ml override: use Sobol-drawn values as-is.
                # When n_ml_per_config=1, the grid formula (m+0.5)/1=0.5 would
                # force ALL proposals to ml=midpoint regardless of Sobol.
                rows_unit.append(base_unit[k])
            else:
                for m in range(self.n_ml_per_config):
                    r = base_unit[k].copy()
                    r[ml_free_j] = (m + 0.5) / self.n_ml_per_config
                    rows_unit.append(r)
        rows_unit = np.array(rows_unit[:self.batch_size])
        raw_free = rows_unit * span + lo_raw
        return self._raw_free_matrix_to_model_list(raw_free)

    def specific_generate_method(self, **kwargs):
        """Propose the next batch of models.

        Warm-up: while fewer than n_initial_random completed-and-valid models
        exist (covers the empty-table and double-call-on-iter-0 cases where
        new rows have all_done=False / NaN chi2), return Sobol random proposals.
        Otherwise fit a GP and maximize the acquisition function (Task 5).
        """
        table = self.current_models.table
        if len(table) == 0:
            n_valid = 0
        else:
            done = np.asarray(table['all_done'], dtype=bool)
            finite = np.isfinite(np.asarray(table[self.chi2], dtype=float))
            n_valid = int(np.sum(done & finite))

        if n_valid < self.n_initial_random:
            self._gp_model = None
            self._last_acq_value = None
            self.model_list = self._propose_random_batch()
            return

        self.model_list = self._gp_acquisition_batch()

    # --- triaxiality constraints (normalized space) --------------------

    def _make_triaxiality_constraints(self):
        """Return (nonlinear_constraints, linear_constraints) over NORMALIZED
        free-parameter coordinates, or (None, None) if not applicable.

        Constraint callables receive a 1D tensor x of shape (n_free,) —
        index with x[idx], never x[..., idx]. Each returns >= 0 when feasible.
        Unnormalization to raw (q,p,u) happens inside the callable.
        """
        if self.qobs is None or not all(k in self._free_qpu_idx
                                        for k in ('q', 'p', 'u')):
            return None, None
        import torch
        lo_raw, hi_raw = self._norm_bounds_arrays()
        lo_t = torch.tensor(lo_raw, dtype=torch.double)
        span_t = torch.tensor(hi_raw - lo_raw, dtype=torch.double)
        jq = self._free_qpu_idx['q']
        jp = self._free_qpu_idx['p']
        ju = self._free_qpu_idx['u']
        qobs = float(self.qobs)

        def _raw(x, j):
            return lo_t[j] + x[j] * span_t[j]

        def c_p_ge_q(x):
            return _raw(x, jp) - _raw(x, jq)           # p - q >= 0

        def c_u_lower(x):
            p_r = _raw(x, jp)
            q_r = _raw(x, jq)
            u_r = _raw(x, ju)
            return u_r - torch.maximum(q_r / qobs, p_r) # u - max(q/qobs,p) >= 0

        def c_u_upper(x):
            p_r = _raw(x, jp)
            u_r = _raw(x, ju)
            upper = torch.clamp(p_r / qobs, max=1.0)
            return upper - u_r                          # min(p/qobs,1) - u >= 0

        nonlinear = [(c_p_ge_q, True), (c_u_lower, True), (c_u_upper, True)]
        return nonlinear, None

    def _feasible_ic_generator(self, acq_function, bounds, q, num_restarts,
                               raw_samples, fixed_features=None, options=None,
                               inequality_constraints=None,
                               equality_constraints=None, **kwargs):
        """Return (num_restarts, q, d) initial conditions, ALL feasible.

        BoTorch 0.18.1 validates that every IC satisfies constraint >= -1e-8
        before optimization; this generator uses rejection sampling to ensure
        all returned points pass.
        """
        import torch
        d = bounds.shape[1]
        lo = bounds[0]
        hi = bounds[1]
        need = num_restarts * q
        collected = []
        attempts = 0
        max_attempts = 50

        nonlinear, _ = self._make_triaxiality_constraints()

        while len(collected) < need and attempts < max_attempts:
            attempts += 1
            n_try = max(need * 30, 512)
            unit = torch.rand(n_try, d, dtype=bounds.dtype)
            unit_np = self._project_unit_to_feasible_qpu(unit.numpy())
            unit = torch.tensor(unit_np, dtype=bounds.dtype)
            cand = lo + (hi - lo) * unit

            for i in range(cand.shape[0]):
                x = cand[i]
                feasible = True
                if nonlinear is not None:
                    for fn, is_intra in nonlinear:
                        if is_intra:
                            if fn(x).item() < -1e-8:
                                feasible = False
                                break
                if feasible:
                    collected.append(x)
                    if len(collected) >= need:
                        break

        if len(collected) < need:
            while len(collected) < need:
                collected.append(collected[len(collected) % max(1, len(collected))])

        result = torch.stack(collected[:need]).reshape(num_restarts, q, d)
        return result

    def _gp_acquisition_batch(self):
        """Fit a GP and maximize qLogEI to produce a batch of models."""
        import torch
        from botorch.models import SingleTaskGP, SingleTaskVariationalGP
        from botorch.fit import fit_gpytorch_mll
        from botorch.acquisition import qLogExpectedImprovement
        from botorch.optim import optimize_acqf
        from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO

        table = self.current_models.table
        X_norm, y, names, lo_raw, hi_raw = extract_gp_training_data(
            table, self.par_space, which_chi2=self.chi2)

        assert names == self.free_param_names, \
            f'param order mismatch: {names} vs {self.free_param_names}'

        X_t = torch.tensor(X_norm, dtype=torch.double)
        chi2_t = torch.tensor(y, dtype=torch.double).unsqueeze(-1)
        Y_t = -chi2_t                        # BoTorch maximizes; negate chi2

        n_train = X_t.shape[0]
        if n_train > 300:
            model = SingleTaskVariationalGP(X_t, Y_t).to(torch.double)
            mll = VariationalELBO(model.likelihood, model.model,
                                  num_data=n_train)
        else:
            model = SingleTaskGP(X_t, Y_t).to(torch.double)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        self._gp_model = model

        d = len(self.free_par_idx)
        bounds = torch.stack([
            torch.zeros(d, dtype=torch.double),
            torch.ones(d, dtype=torch.double)])

        acqf = qLogExpectedImprovement(model=model, best_f=Y_t.max())

        nonlinear, linear = self._make_triaxiality_constraints()
        opt_kwargs = dict(
            acq_function=acqf, bounds=bounds, q=self.batch_size,
            num_restarts=10, raw_samples=128)
        if nonlinear is not None:
            opt_kwargs['nonlinear_inequality_constraints'] = nonlinear
            opt_kwargs['ic_generator'] = self._feasible_ic_generator
            opt_kwargs['options'] = {'batch_limit': 1}

        candidates, acq_value = optimize_acqf(**opt_kwargs)
        self._last_acq_value = float(acq_value.item())

        cand_np = candidates.detach().numpy()
        raw_free = denormalize_to_raw(cand_np, lo_raw, hi_raw)
        return self._raw_free_matrix_to_model_list(raw_free)

    def check_specific_stopping_criteria(self):
        """BayesOpt convergence signals plus the inherited chi2 backstop.

        Calls super() first (the chi2-plateau backstop). Then, if a GP has
        been fitted, checks (1) max posterior variance over a 256-point Sobol
        grid and (2) the last qLogEI batch value.
        """
        super().check_specific_stopping_criteria()

        if self._gp_model is None:
            return

        import torch
        from torch.quasirandom import SobolEngine

        d = len(self.free_par_idx)
        sobol = SobolEngine(dimension=d, scramble=True)
        X_grid = sobol.draw(256).to(dtype=torch.double)
        with torch.no_grad():
            posterior = self._gp_model.posterior(X_grid)
            max_variance = posterior.variance.max().item()

        self.status['gp_max_variance_low'] = (
            self.max_gp_variance_threshold is not None
            and max_variance < self.max_gp_variance_threshold)

        if (self._last_acq_value is not None
                and self.min_ei_threshold is not None):
            self.status['gp_min_ei_low'] = (
                self._last_acq_value < self.min_ei_threshold)
        else:
            self.status['gp_min_ei_low'] = False


class FullGrid(ParameterGenerator):
    """
    A full cartesian grid.

    A full Cartesian grid in all free parameters, with bounds ``lo/hi`` and
    stepsize ``step``. **Warning**: If several (>3) parameters are free, this
    will result in a large number of models. This parameter generator is
    generally not intended for production use.

    Parameters
    ----------
    par_space : ``dyn.parameter_space.ParameterSpace`` object
    parspace_settings : dict

    """
    def __init__(self,
                 par_space=[],
                 parspace_settings=None):
        super().__init__(par_space=par_space,
                         parspace_settings=parspace_settings,
                         name='FullGrid')
        self.logger = logging.getLogger(f'{__name__}.{__class__.__name__}')
        self.step = []
        self.minstep = []
        try:
            for par in self.par_space:
                settings = par.par_generator_settings
                if par.fixed is False:
                    self.step.append(settings['step'])
                    # use 'minstep' value if present, otherwise use 'step'
                    self.minstep.append(settings['minstep'] \
                        if 'minstep' in settings else settings['step'])
                else:
                    self.step.append(None)
                    self.minstep.append(None)
        except:
            text = 'FullGrid: non-fixed parameters need step setting'
            self.logger.error(text)
            raise ValueError(text)

        stop_crit = parspace_settings['stopping_criteria']
        stop_abs = 'min_delta_chi2_abs' in stop_crit
        stop_rel = 'min_delta_chi2_rel' in stop_crit
        if (stop_abs and stop_rel) or not (stop_abs or stop_rel):
            text = 'FullGrid: specify exactly one of the ' + \
                   'options min_delta_chi2_abs, min_delta_chi2_rel'
            self.logger.error(text)
            raise ValueError(text)
        if stop_abs:
            self.min_delta_chi2_abs = stop_crit['min_delta_chi2_abs']
        else:
            self.min_delta_chi2_abs = None
        if stop_rel:
            self.min_delta_chi2_rel = stop_crit['min_delta_chi2_rel']
        else:
            self.min_delta_chi2_rel = None

    def specific_generate_method(self, **kwargs):
        """
        Generates new models

        Span the whole parameter grid.
        The center of the grid walk is the parameter set with the smallest chi2
        value, depending on the parameter space setting 'which_chi2'.

        Parameters
        ----------
        None.

        Raises
        ------
        None.

        Returns
        -------
        None. self.model_list is the list of new models.
        """
        if len(self.current_models.table) == 0:
            # The 'zeroth iteration' results in only one model
            # (all parameters at their .raw_value level)
            self.model_list = [[p for p in self.par_space]]
        else: # Subsequent iterations...
            if len(self.current_models.table) == 1: # 'first' iteration
                center_idx = 0
            else:
                # center criterion: min(chi2)
                center_idx = np.nanargmin(self.current_models.table[self.chi2])
            n_par = self.par_space.n_par
            center = list(self.current_models.table[center_idx])[:n_par]
            raw_center = self.par_space.get_raw_value_from_param_value(center)
            self.logger.debug(f'center: {center}')
            # Build model_list by walking the grid
            self.model_list = []
            self.grid(center=raw_center)
            # for m in self.model_list:
            #     self.logger.debug(f'{[(p.name, p.raw_value) for p in m]}')

    def grid(self, center=None, par=None, eps=1e-6):
        """
        Create the grid

        Span the whole parameter grid defined by
        ``self.par_space.par_generator_settings`` attributes.
        IN GENERAL THIS WILL RESULT IN A LARGE NUMBER OF MODELS ADDED TO
        self.model_list! PRIMARILY THIS IS INTENDED FOR TESTING AND DEBUGGING.
        Clips parameter values to lo/hi attributes. If clipping violates the
        minstep attribute, the resulting model(s) will not be created. If the
        minstep attribute is missing, the step attribute will be used instead.
        Explicitly set minstep=0 to allow arbitrarily small steps down to eps
        (not recommended).

        Parameters
        ----------
        center : List of center coordinates. Must be in the same sequence as
                 the parameters in self.par_space. Mandatory argument.
        par : Internal use only. Gives the parameter to start with. Set
              automatically in the recursive process. The default is None.
        eps : Used for numerical comparison (relative tolerance), default 1e-6

        Raises
        ------
        ValueError if center is not specified or fixed parameters != center.

        Returns
        -------
        None. Sets self.model_list to the resulting models.

        """
        if center is None:
            text = 'Need center'
            self.logger.error(text)
            raise ValueError(text)
        if not par:
            par = self.par_space[0]
        paridx = self.par_space.index(par)
        self.logger.debug(f'Call with paridx={paridx}, '
                          f'n_par={self.par_space.n_par}')

        if par.fixed:
            raw_values = [par.raw_value]
            if abs(center[paridx] - par.raw_value) > eps:
                text='Something is wrong: fixed parameter value not in center'
                self.logger.error(text)
                raise ValueError(text)
        else:
            lo = self.lo[paridx]
            hi = self.hi[paridx]
            step = self.step[paridx]
            minstep = self.minstep[paridx]
            # up to 3 *distinct* par_raw (clipped lo, mid, hi values)
            raw_values = []
            raw_value = center[paridx]
            # add the center
            raw_values.append(self.clip(raw_value, lo, hi))
            # start with lo...
            while raw_value >= lo:
                raw_new = self.clip(raw_value-step, lo, hi)
                if abs(raw_value-raw_new) >= max(minstep,eps) \
                                             - sys.float_info.epsilon:
                    raw_values.append(raw_new)
                else:
                    break
                raw_value = raw_new
            # now hi...
            raw_value = center[paridx]
            while raw_value <= hi:
                raw_new = self.clip(raw_value+step, lo, hi)
                if abs(raw_value-raw_new) >= max(minstep,eps) \
                                             - sys.float_info.epsilon:
                    raw_values.append(raw_new)
                else:
                    break
                raw_value = raw_new

        for raw_value in raw_values:
            parcpy = copy.deepcopy(par)
            parcpy.raw_value = raw_value
            if not self.model_list: # add first entry if model_list is empty
                self.model_list = [[parcpy]]
                models_prev = [[]]
                self.logger.debug('new model list, starting w/parameter '
                                  f'{parcpy.name}')
            elif parcpy.name in [p.name for p in self.model_list[0]]:
                # in this case, create new (partial) model by copying last
                # models and setting the new parameter raw_value
                for m in models_prev:
                    new_model = m + [parcpy]
                    self.model_list.append(new_model)
                self.logger.debug(f'{parcpy.name} is in '
                      f'{[p.name for p in self.model_list[0]]}, '
                      f'added {parcpy.name}={parcpy.raw_value}')
            else: # new parameter: append it to existing (partial) models
                models_prev = copy.deepcopy(self.model_list)
                for m in self.model_list:
                    m.append(parcpy)
                self.logger.debug(f'new parameter {parcpy.name}='
                                  f'{parcpy.raw_value}')

        # call recursively until all paramaters are done:
        if paridx < self.par_space.n_par - 1:
            self.grid(center=center, par=self.par_space[paridx+1])


class SpecificModels(ParameterGenerator):
    """
    Create specific models.

    Creates models with parameter values according to the entries in the
    lists ``specific_values`` in a single iteration. If any parameter's
    ``specific_values`` entry is missing, its ``value`` entry will be used.
    ``parspace_settings['generator_settings']['SpecificModels_mode']``
    determines how models are constructed:
    ``list``: selects parameter values element-wise. All parameters'
    ``specific_values`` lists must be of equal length (or zero length if their
    respective ``value`` entry is to be used).
    ``cartesian``: Cartesian product of specific parameter values. The
    parameters' ``specific_values`` lists don't need to be of equal length.
    May result in a large number of models.

    Note that this parameter generator ignores ``lo``, ``high``, ``step``,
    ``minstep``, and ``fixed``. Also, ``value`` will be ignored if
    ``specific_values`` is specified.

    Further, all models are run in a single iteration and the optimality
    tolerances in the ``stopping_criteria`` section in the configuration file's
    ``parameter_space_settings`` will be ignored.

    Parameters
    ----------
    par_space : ``dyn.parameter_space.ParameterSpace`` object
    parspace_settings : dict

    """
    def __init__(self,
                 par_space=[],
                 parspace_settings=None):
        super().__init__(par_space=par_space,
                         parspace_settings=parspace_settings,
                         name='SpecificModels')
        self.logger = logging.getLogger(f'{__name__}.{__class__.__name__}')
        try:
            self.mode = self.parspace_settings['generator_settings']\
                                              ['SpecificModels_mode'].lower()
        except:
            text = 'Need SpecificModels_mode setting in generator_settings.'
            self.logger.error(text)
            raise ValueError(text)
        if self.mode not in ('list', 'cartesian'):
            text = 'Mode must either be "list" or "cartesian".'
            self.logger.error(text)
            raise ValueError(text)

    def specific_generate_method(self, **kwargs):
        """
        Generates the specific models

        Parameters
        ----------
        None.

        Returns
        -------
        None.
            sets ``self.model_list``, the list of new models.

        """
        self.model_list = []
        par_list_idx = \
          [i for i in range(len(self.par_space))
             if self.par_space[i].par_generator_settings
             if 'specific_values' in self.par_space[i].par_generator_settings]
        if len(par_list_idx) == 0: # nothing to do really...
            self.model_list.append([copy.deepcopy(p) for p in self.par_space])
            self.logger.info('Found ONE individual model.')
            return ###########################################################

        lengths = \
            [len(self.par_space[i].par_generator_settings['specific_values'])
             for i in par_list_idx]
        if self.mode == 'list':
            if len(set(lengths)) > 1:
                text = 'For a simple list of new models all specific_values ' \
                       'lists must be of equal length.'
                self.logger.error(text)
                raise ValueError(text)
            n_mod = lengths[0]
        else:
            n_mod = np.prod(lengths)
        self.logger.info(f'Adding {n_mod} individual models.')

        specific_values = \
            [self.par_space[i].par_generator_settings['specific_values']
             for i in par_list_idx]
        if self.mode == 'list':
            for i in range(n_mod):
                new_parset = [copy.deepcopy(p) for p in self.par_space]
                for val_idx, idx in enumerate(par_list_idx):
                    new_parset[idx].raw_value = specific_values[val_idx][i]
                self.model_list.append([copy.deepcopy(p) for p in new_parset])
        else:
            for val in itertools.product(*specific_values):
                new_parset = [copy.deepcopy(p) for p in self.par_space]
                for val_idx, idx in enumerate(par_list_idx):
                    new_parset[idx].raw_value = val[val_idx]
                self.model_list.append([copy.deepcopy(p) for p in new_parset])

        return

    def check_specific_stopping_criteria(self):
        """The specific stopping criteria

        Will always stop after creating all specific models.

        Returns
        -------
        None
            Sets ``self.status['min_delta_chi2_reached']`` to ``True``

        """
        self.status['min_delta_chi2_reached'] = True

# end
