# classes to hold the physical components of the system
# e.g. the stellar light, dark matter, black hole, globular clusters

import numpy as np
from scipy import special, integrate

import logging

import dynamite as dyn
from dynamite import mges as mge

class System(object):
    """The physical system being modelled

    e.g. system is a galaxy. A system is composed of ``Components`` e.g. the
    galaxy is composed of stars, black hole, dark matter halo. This object is
    automatically created when the configuration file is read.
    """
    def __init__(self, *args):
        self.logger = logging.getLogger(f'{__name__}.{__class__.__name__}')
        self.n_cmp = 0
        self.cmp_list = []
        self.n_kin = 0
        self.n_pop = 0
        self.parameters = None
        self.distMPc = None
        self.name = None
        for component in args:
            self.add_component(component)

    def add_component(self, cmp):
        """add a component to the system

        Parameters
        ----------
        cmp : a ``dyn.physical_system.Component`` object

        Returns
        -------
        None
            updated the system componenent attributes

        """
        self.cmp_list += [cmp]
        self.n_cmp += 1
        self.n_kin += len(cmp.kinematic_data)
        self.n_pop += len(cmp.population_data)

    def validate(self):
        """
        Validate the system

        Ensures the System has the required attributes: at least one component,
        no duplicate component names, and the ml parameter, and that the
        sformat string for the ml parameter is set. Also validates
        the parameter generator settings' minstep value for ml if it is a
        non-fixed parameter.

        Raises
        ------
        ValueError : if required attributes or components are missing, or if
                     there is no ml parameter

        Returns
        -------
        None.

        """
        if len(self.cmp_list) != len(set(self.cmp_list)):
            raise ValueError('No duplicate component names allowed')
        if (self.distMPc is None) or (self.name is None):
            text = 'System needs distMPc and name attributes'
            self.logger.error(text)
            raise ValueError(text)
        if not self.cmp_list:
            text = 'System has no components'
            self.logger.error(text)
            raise ValueError(text)
        #if len(self.parameters) != 1 and self.parameters[0].name != 'ml':
        if self.parameters[0].name != 'ml':
            text = 'System needs ml as its first parameter.'
            self.logger.error(text)
            raise ValueError(text)
        ml = self.parameters[0]
        ml.update(sformat = '05.2f') # sformat of ml parameter
        if not ml.fixed and 'minstep' in ml.par_generator_settings:
            generator_settings = ml.par_generator_settings
            if generator_settings['minstep'] > generator_settings['step']:
                text = f"{self.__class__.__name__} parameter {ml.name}'s " \
                       "parameter generator settings have minstep > step, " \
                       f"setting minstep=step={generator_settings['step']}."
                self.logger.warning(text)
                generator_settings['minstep'] = generator_settings['step']
        if len(self.parameters) > 1:
            # omega = self.parameters[1]
            if self.parameters[1].name != 'omega':
                text = 'System needs omega as its second parameter.'
                self.logger.error(text)
                raise ValueError(text)
        if len(self.parameters) > 2:
            text = 'System can only have ml and omega parameters, not ' \
                   f'{[p.name for p in self.parameters]} - check for typos.'
            self.logger.error(text)
            raise ValueError(text)

    def validate_parset(self, par):
        """
        Validates the system's parameter values

        Kept separate from the validate method to facilitate easy calling from
        the ``ParameterGenerator`` class. Returns `True` if all parameters are
        non-negative, except for logarithmic parameters which are not checked.

        Parameters
        ----------
        par : dict
            { "p":val, ... } where "p" are the system's parameters and
            val are their respective raw values

        Returns
        -------
        isvalid : bool
            True if the parameter set is valid, False otherwise

        """
        p_raw_values = [par[p.name]
                        for p in self.parameters if not p.logarithmic]
        isvalid = np.all(np.sign(p_raw_values) >= 0)
        if not isvalid:
            self.logger.debug(f'Invalid system parameters {par}: at least '
                              'one negative non-log parameter.')
        return bool(isvalid)

    def get_par_by_name(self, n):
        """
        Get a parameter using its name.

        Parameters
        ----------
        n : str
            The parameter name.

        Returns
        -------
        p : a ``dyn.parameter_space.Parameter`` object
            The parameter in question.

        """
        ps = self.parameters
        return ps[[p.name for p in ps].index(n)]

    def __repr__(self):
        return f'{self.__class__.__name__} with {self.__dict__}'

    def get_component_from_name(self, cmp_name):
        """get_component_from_name

        Parameters
        ----------
        cmp_name : string
            component name (as specified in the congi file)

        Returns
        -------
        a ``dyn.physical_system.Component`` object

        """
        cmp_list_list = np.array([cmp0.name for cmp0 in self.cmp_list])
        idx = np.where(cmp_list_list == cmp_name)
        self.logger.debug(f'Checking for 1 and only 1 component {cmp_name}...')
        error_msg = f"There should be 1 and only 1 component named {cmp_name}"
        assert len(idx[0]) == 1, error_msg
        self.logger.debug('...check ok.')
        component = self.cmp_list[idx[0][0]]
        return component

    def get_component_from_class(self, cmp_class):
        """get_component_from_class

        Parameters
        ----------
        cmp_class : string
            name of the component type/class

        Raises
        ------
        ValueError : if there are more than one component of the same class.
            # TODO: remove this limit, e.g. if we had two MGE-based components
            one for stars, one for gas

        Returns
        -------
        a ``dyn.physical_system.Component`` object

        """
        self.logger.debug('Checking for 1 and only 1 component of class '
                          f'{cmp_class}...')
        components = filter(lambda c: isinstance(c,cmp_class), self.cmp_list)
        component = next(components, False)
        if component is False or next(components, False) is not False:
            error_msg = 'Actually... there should be 1 and only 1 ' \
                        f'component of class {cmp_class}'
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        self.logger.debug('...check ok.')
        return component

    def get_all_mge_components(self):
        """Get all components which contain MGEs

        Returns
        -------
        list
            a list of Component objects

        """
        mge_cmp = [c for c in self.cmp_list
                   if isinstance(c, TriaxialVisibleComponent)
                   or isinstance(c, BarDiskComponent)]
        return mge_cmp

    def get_unique_triaxial_visible_component(self):
        """Return the unique non-bar MGE component (raises an error if there
        are zero or multiple such components)

        Returns
        -------
            a ``dyn.physical_system.TriaxialVisibleComponent`` object

        """
        mges = self.get_all_mge_components()
        if len(mges) > 1:
            self.logger.error('Found more than one triaxial visible component')
            raise ValueError('Found more than one triaxial visible component')
        if len(mges) == 0:
            self.logger.error('Found zero triaxial visible components')
            raise ValueError('Found zero triaxial visible components')
        return mges[0]

    def get_all_bar_components(self):
        """Get all components which are rotating MGEs (i.e. bars)

        Returns
        -------
        list
            list of Component objects, keeping only the rotating MGE components

        """
        bar_cmp = [c for c in self.cmp_list if isinstance(c, BarDiskComponent)]
        return bar_cmp

    def get_unique_bar_component(self):
        """Return the unique rotating bar component (raises an error if there
        are zero or multiple such components)

        Returns
        -------
            a ``dyn.physical_system.BarDiskComponent`` object

        """
        bars = self.get_all_bar_components()
        if len(bars) > 1:
            self.logger.error('Found more than one bar component')
            raise ValueError('Found more than one bar component')
        if len(bars) == 0:
            self.logger.error('Found zero bar components')
            raise ValueError('Found zero bar components')
        return bars[0]

    def get_all_dark_components(self):
        """Get all components which are Dark

        Returns
        -------
        list
            a list of Component objects, keeping only the dark components

        """
        dark_cmp = [c for c in self.cmp_list if isinstance(c, DarkComponent)]
        return dark_cmp

    def get_all_dark_non_plummer_components(self):
        """Get all Dark components which are not plummer

        Useful in legacy orbit libraries for finding the dark halo component.
        For legacy models, the black hole is always a plummer, so any Dark but
        non plummer components must represent the dark halo.

        Returns
        -------
        list
            a list of Component objects, keeping only the dark components

        """
        dark_cmp = self.get_all_dark_components()
        dark_non_plum_cmp = [c for c in dark_cmp if not isinstance(c, Plummer)]
        return dark_non_plum_cmp

    def get_sbh_component(self):
        """Get the stellar-black-hole component, if any

        Returns
        -------
        StellarBlackHoles or StellarBlackHolesMGE or None

        Raises
        ------
        ValueError : if more than one sBH component is present

        """
        sbh = [c for c in self.cmp_list
               if isinstance(c, (StellarBlackHoles, StellarBlackHolesMGE))]
        if len(sbh) > 1:
            text = f'System can have at most one sBH component, not {len(sbh)}'
            self.logger.error(text)
            raise ValueError(text)
        return sbh[0] if sbh else None

    def get_halo_component(self):
        """Get the dark halo component, if any

        The halo is any dark, non-Plummer, non-sBH component.

        Returns
        -------
        Component or None

        Raises
        ------
        ValueError : if more than one halo is present

        """
        halo = [c for c in self.get_all_dark_non_plummer_components()
                if not isinstance(c, (StellarBlackHoles,
                                      StellarBlackHolesMGE))]
        if len(halo) > 1:
            text = f'System can have at most one DM halo, not {len(halo)}'
            self.logger.error(text)
            raise ValueError(text)
        return halo[0] if halo else None

    def get_unique_ext_chi2_component(self):
        """Return the unique Chi2Ext component

        Raises
        ------
        ValueError
            If there are multiple Chi2Ext components

        Returns
        -------
            a ``dyn.physical_system.Chi2Ext`` object

        """
        chi2_ext_cmp = [c for c in self.cmp_list if isinstance(c, Chi2Ext)]
        if len(chi2_ext_cmp) > 1:
            self.logger.error('Found more than one Chi2Ext component')
            raise ValueError('Found more than one Chi2Ext component')
        if len(chi2_ext_cmp) == 0:
            return None
        else:
            return chi2_ext_cmp[0]

    @property
    def has_chi2_ext(self):
        """True if the system has an external chi2 component, else False
        """
        return False if self.get_unique_ext_chi2_component() is None else True

    def get_all_kinematic_data(self):
        """get_all_kinematic_data

        Loop over all components, extract their kinemtics into a list.

        Returns
        -------
        list
            all_kinematics in a list

        """
        all_kinematics = []
        for component in self.cmp_list:
            all_kinematics += component.kinematic_data
        return all_kinematics

    def is_bar_disk_system(self):
        """is_bar_disk_system

        Check if the system contains at least one bar component and at least
        one disk component.

        Returns
        -------
        isbardisk : Bool
            System contains a bar and a disk.
        """
        isbardisk = len(self.get_all_bar_components()) > 0
        return isbardisk

    def number_of_visible_components(self):
        return sum(1 for i in self.cmp_list if isinstance(i, VisibleComponent))

    def number_of_bar_components(self):
        return sum(1 for i in self.cmp_list if isinstance(i, BarDiskComponent))

class Component(object):
    """A component of the physical system

    e.g. the stellar component, black hole, or dark halo of a galaxy

    Parameters
    ----------
    name : string
        a short but descriptive name of the component
    visible : Bool
        whether this is visible <--> whether it has an associated MGE
    symmetry : string
        one of 'spherical', 'axisymm', or 'triax' **not currently used**
    kinematic_data : list
        a list of ``dyn.kinemtics.Kinematic`` data for this component
    parameters  : list
        a list of ``dyn.parameter_space.Parameter`` objects for this component
    population_data : list
        a list of ``dyn.populations.Population`` data for this component **not
        currently used**

    """
    def __init__(self,
                 name = None,
                 visible=None,
                 symmetry=None,
                 kinematic_data=[],
                 population_data=[],
                 parameters=[]):
        self.logger = logging.getLogger(f'{__name__}.{__class__.__name__}')
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        self.visible = visible
        self.symmetry = symmetry
        self.kinematic_data = kinematic_data
        self.population_data = population_data
        self.parameters = parameters

    def validate(self, par=None):
        """
        Validate the component

        Ensure it has the required attributes and parameters. Also validates
        the parameter generator settings' minstep value for non-fixed
        parameters.

        Parameters
        ----------
        par : a list with parameter names. Mandatory.

        Raises
        ------
        ValueError : if a required attribute is missing or the required
                     parameters do not exist

        Returns
        -------
        None.

        """
        errstr = f'Component {self.__class__.__name__} needs attribute '
        if self.visible is None:
            text = errstr + 'visible'
            self.logger.error(text)
            raise ValueError(text)
        if not self.parameters:
            text = errstr + 'parameters'
            self.logger.error(text)
            raise ValueError(text)

        pars = [self.get_parname(p.name) for p in self.parameters]
        if set(pars) != set(par):
            text = f'{self.__class__.__name__} needs parameter(s) ' + \
                   f'{par}, not {pars}.'
            self.logger.error(text)
            raise ValueError(text)

        for p in [p for p in self.parameters
                  if not p.fixed and 'minstep' in p.par_generator_settings]:
            generator_settings = p.par_generator_settings
            if generator_settings['minstep'] > generator_settings['step']:
                text = f"{self.__class__.__name__} parameter {p.name}'s " \
                       "parameter generator settings have minstep > step, " \
                       f"setting minstep=step={generator_settings['step']}."
                self.logger.warning(text)
                generator_settings['minstep'] = generator_settings['step']

    def validate_parset(self, par):
        """
        Validates the component's parameter values.

        Kept separate from the
        validate method to facilitate easy calling from the parameter
        generator class. This is a `placeholder` method which returns
        `True` if all parameters are non-negative, except for logarithmic
        parameters which are not checked. Specific validation
        should be implemented for each component subclass.

        Parameters
        ----------
        par : dict
            { "p":val, ... } where "p" are the component's parameters and
            val are their respective raw values

        Returns
        -------
        isvalid : bool
            True if the parameter set is valid, False otherwise

        """
        p_raw_values = [par[self.get_parname(p.name)]
                    for p in self.parameters if not p.logarithmic]
        isvalid = np.all(np.sign(p_raw_values) >= 0)
        if not isvalid:
            self.logger.debug(f'Invalid parameters {par}: at least one '
                              'negative non-log parameter.')
        return isvalid

    def get_parname(self, par):
        """
        Strip the component name suffix from the parameter name.

        Parameters
        ----------
        par : str
            The full parameter name "parameter-component".

        Returns
        -------
        pure_parname : str
            The parameter name without the component name suffix.

        """
        try:
            pure_parname = par[:par.rindex(f'-{self.name}')]
        except:
            self.logger.error(f'Component name {self.name} not found in '
                              f'parameter string {par}')
            raise
        return pure_parname

    def get_par_by_name(self, n):
        """
        Get a parameter using its (unsuffixed) name.

        Parameters
        ----------
        n : str
            The parameter name (without the component name suffix)

        Returns
        -------
        p : a ``dyn.parameter_space.Parameter`` object
            The parameter in question.

        """
        ps = self.parameters
        return ps[[p.name for p in ps].index(n + '-' + self.name)]

    def __repr__(self):
        return (f'\n{self.__class__.__name__}({self.__dict__}\n)')


class VisibleComponent(Component):
    """Any visible component of the sytem, with an MGE

    Parameters
    ----------
    mge_pot : a ``dyn.mges.MGE`` object
        describing the (projected) surface-mass density
    mge_lum : a ``dyn.mges.MGE`` object
        describing the (projected) surface-luminosity density

    """
    def __init__(self,
                 mge_pot=None,
                 mge_lum=None,
                 **kwds):
         # visible components have MGE surface density
        self.mge_pot = mge_pot
        self.mge_lum = mge_lum
        self.mass_aper = None
        super().__init__(visible=True, **kwds)
        self.logger = logging.getLogger(f'{__name__}.{__class__.__name__}')

    def get_M_stars_tot(self, distance, parset):
        """
        Calculates and returns the total stellar mass via the mge.

        Parameters
        ----------
        distance : float
            Distance of the system in MPc
        parset : astropy table row
            must contain mass-to-light ratio ml

        Returns
        -------
        float
            Total stellar mass

        """
        mgepar = self.mge_pot.data
        mgeI = mgepar['I']
        mgesigma = mgepar['sigma']
        mgeq = mgepar['q']

        arctpc = distance*np.pi/0.648
        sigobs_pc = mgesigma*arctpc

        return 2 * np.pi * np.sum(mgeI * mgeq * sigobs_pc ** 2) * parset['ml']

    def validate(self, **kwds):
        super().validate(**kwds)
        if not (isinstance(self.mge_pot, mge.MGE) and \
                isinstance(self.mge_lum, mge.MGE)):
            text = f'{self.__class__.__name__}.mge_pot and ' \
                   f'{self.__class__.__name__}.mge_lum ' \
                    'must be mges.MGE objects'
            self.logger.error(text)
            raise ValueError(text)


class AxisymmetricVisibleComponent(VisibleComponent):

    def __init__(self, **kwds):
        super().__init__(symmetry='axisymm', **kwds)

    def validate(self):
        par = ['par1', 'par2']
        super().validate(par=par)


class TriaxialVisibleComponent(VisibleComponent):
    """Triaxial component with a MGE projected density

    Has parameters (p,q,u) = (b/a, c/a, sigma_obs/sigma_intrinsic) used for
    deprojecting the MGE. A given (p,q,u) correspond to a fixed set of
    `viewing angles` for the triaxial ellipsoid.

    """
    def __init__(self, **kwds):
        super().__init__(symmetry='triax', **kwds)
        self.logger = logging.getLogger(f'{__name__}.{__class__.__name__}')
        self.qobs = np.nan
        self.par = ['q', 'p', 'u']

    def validate(self):
        """
        Validate the TriaxialVisibleComponent

        Validates parameter names and sets self.qobs
        (minimal flattening from mge data).

        Returns
        -------
        None.

        """
        super().validate(par=self.par)
        self.qobs = np.amin(self.mge_pot.data['q'])
        if np.isnan(self.qobs):
            raise ValueError(f'{self.__class__.__name__}.qobs is np.nan')

    def validate_parset(self, par):
        """
        Validate the p, q, u parameters

        Validates the triaxial component's p, q, u parameters. Requires
        self.qobs to be set. A parameter set is valid if the resulting
        (theta, psi, phi) are not np.nan.

        Parameters
        ----------
        par : dict
            { "p":val, ... } where "p" are the component's parameters and
            val are their respective values

        Returns
        -------
        bool
            True if the parameter set is valid, False otherwise

        """
        tpp = self.triax_pqu2tpp(par['p'], par['q'], par['u'])
        return bool(not np.any(np.isnan(tpp)))

    def triax_pqu2tpp(self,p,q,u):
        """
        transform axis ratios to viewing angles

        transfer (p, q, u) to the three viewing angles (theta, psi, phi)
        with known flatting self.qobs.
        Taken from schw_basics, same as in vdB et al. 2008, MNRAS 385,2,647
        We should possibly revisit the expressions later

        """
        # avoid legacy_fortran's u=1 (rather, phi=psi=90deg) problem
        if u == 1:
            u *= (1-np.finfo(float).epsneg)  # same value as for np.double
        p2 = np.double(p) ** 2
        q2 = np.double(q) ** 2
        u2 = np.double(u) ** 2
        o2 = np.double(self.qobs) ** 2
        # Check for p=0
        if np.isclose(p, 0.):
            theta = phi = psi = np.nan
            self.logger.debug('DEPROJ FAIL: p=0')
        # Check for q=0
        if np.isclose(q, 0.):
            theta = phi = psi = np.nan
            self.logger.debug('DEPROJ FAIL: q=0')
        # Check for u=0
        if np.isclose(u, 0.):
            theta = phi = psi = np.nan
            self.logger.debug('DEPROJ FAIL: u=0')
        # Check for u>1
        if u>1:
            theta = phi = psi = np.nan
            self.logger.debug('DEPROJ FAIL: u>1')
        if np.isclose(u,p):
            u=p
        # Check for possible triaxial deprojection (v. d. Bosch 2004,
        # triaxpotent.f90 and v. d. Bosch et al. 2008, MNRAS 385, 2, 647)
        str = f'{q} <= {p} <= {1}, ' \
              f'{max((q/self.qobs,p))} < {u} <= {min((p/self.qobs),1)}, ' \
              f'q\'={self.qobs}'
        # 0<=t<=1, t = (1-p2)/(1-q2) and p,q>0 is the same as 0<q<=p<=1 and q<1
        t = (1-p2)/(1-q2)
        if not (0 <= t <= 1) or \
           not (max((q/self.qobs,p)) < u <= min((p/self.qobs),1)) :
            theta = phi = psi = np.nan
            self.logger.debug(f'DEPROJ FAIL: {str}')
        else:
            self.logger.debug(f'DEPROJ PASS: {str}')
            w1 = (u2 - q2) * (o2 * u2 - q2) / ((1.0 - q2) * (p2 - q2))
            w2 = (u2 - p2) * (p2 - o2 * u2) * (1.0 - q2) / ((1.0 - u2) * (1.0 - o2 * u2) * (p2 - q2))
            w3 = (1.0 - o2 * u2) * (p2 - o2 * u2) * (u2 - q2) / ((1.0 - u2) * (u2 - p2) * (o2 * u2 - q2))

            if w1 >=0.0 :
                theta = np.arccos(np.sqrt(w1)) * 180 /np.pi
            else:
                theta=np.nan

            if w2 >=0.0 :
                phi = np.arctan(np.sqrt(w2)) * 180 /np.pi
            else:
                phi=np.nan

            if w3 >=0.0 :
                psi = 180 - np.arctan(np.sqrt(w3)) * 180 /np.pi
            else:
                psi=np.nan
        self.logger.debug(f'theta={theta}, phi={phi}, psi={psi}')
        return theta,psi,phi

    def find_grid_of_valid_pqu(self, n_grid=200):
        """Find valid values of the parameters (p,q,u)

        Creates a grid of all values of 0<(p,q,u)<=1, and finds those which have
        a valid deprojection subject to the fulfilment of all three criteria:
        1. 0 < q <= p <=1
        2. max(q/qobs, p) < u
        3. u< min(p/qobs, 1)
        where qobs is the smallest value of q for the MGE.

        Parameters
        ----------
        n_grid : int
            grid size used for p,q,u

        Returns
        -------
        (p,q,u), valid
            3d grids of p,q,u values, and boolen array `valid` which is `True`
            for valid values

        """
        # make grid of possible p,q,u values
        p = np.linspace(0, 1, n_grid)[1:]
        q = np.linspace(0, 1, n_grid)[1:]
        u = np.linspace(0, 1, n_grid)[1:]
        p, q, u = np.meshgrid(p, q, u, indexing='ij')
        # check three conditions for whether (p,q,u) give a valid deprojection
        invalid_a = q>p
        invalid_b = np.maximum(q/self.qobs, p) >= u
        invalid_c = u >= np.minimum(p/self.qobs, 1.)
        # combine the conditions
        invalid_ab = np.logical_or(invalid_a, invalid_b)
        invalid_abc = np.logical_or(invalid_ab, invalid_c)
        valid = np.logical_not(invalid_abc)
        return (p,q,u), valid

    def suggest_parameter_values(self, target_u=0.9):
        """Suggest valid values of the parameters (p,q,u)

        Find valid values using the mehtod `find_grid_of_valid_pqu`. Then for
        each of (p,q,u), we suggest values:
        - lo/hi : the min/max of all valid values
        - value : u=target_u, and p/q = mean of all valid p/q values where u is
        close to target value
        - step/minstep : a fifth/twentieth of the range of valid values

        Parameters
        ----------
        target_u : float
            Desired value of the parameter u

        Returns
        -------
        string
            text to print out suggesting quantities for (p,q,u)
        """
        (p, q, u), valid = self.find_grid_of_valid_pqu()
        text = "No deprojection possible for the specificed values of (p,q,u)."
        text += " Here are some suggestions:\n"
        # take avg of valid p's and q's where u is close to targer value
        target_u = 0.9
        idx = np.where(np.abs(u[valid]-target_u)<0.005)
        if idx[0].shape==(0,):
            text = f"Cannot suggest valid (p,q,u) for a target u={target_u}"
            self.logger.error(text)
            raise ValueError(text)
        suggest_p = np.mean(p[valid][idx])
        suggest_q = np.mean(q[valid][idx])
        suggested_values = [suggest_p, suggest_q, target_u]
        for (symbol, array, val) in zip(['p', 'q', 'u'],
                                        [p[valid], q[valid], u[valid]],
                                        suggested_values):
            lo, hi = np.min(array), np.max(array)
            step = (hi-lo)/5.
            minstep = (hi-lo)/20.
            text += f'\t{symbol}:\n'
            text += f'\t\t lo : {lo:.2f}\n'
            text += f'\t\t hi : {hi:.2f}\n'
            text += f'\t\t step : {step:.2f}\n'
            text += f'\t\t minstep : {minstep:.2f}\n'
            text += f'\t\t value : {val:.2f}\n'
        return text

    @staticmethod
    def triax_tpp2pqu(theta, phi, psi, qobs_pot, psi_off):
        """
        transform viewing angles to axis ratios
        """
        theta_view = np.deg2rad(theta)
        phi_view   = np.deg2rad(phi)
        psi_obs    = np.deg2rad(psi + psi_off)

        costh = np.cos(theta_view)
        tanph = np.tan(phi_view)
        if np.abs(costh) < 1.0e-6:
            print(
                "triax_tpp2pqu: |cos(theta)| too small -> invalid geometry.")
            return np.nan, np.nan, np.nan

        if np.abs(tanph) < 1.0e-6:
            print(
                "triax_tpp2pqu: |tan(phi)| too small -> invalid geometry.")
            return np.nan, np.nan, np.nan

        secth = 1.0 / costh
        cotph = 1.0 / tanph

        delp = 1.0 - qobs_pot ** 2

        nom1minq2 = delp * (
            2.0 * np.cos(2.0 * psi_obs) + np.sin(2.0 * psi_obs) *
            (secth * cotph - np.cos(theta_view) * np.tan(phi_view)))
        nomp2minq2 = delp * (
            2.0 * np.cos(2.0 * psi_obs) + np.sin(2.0 * psi_obs) *
            (np.cos(theta_view) * cotph - secth * np.tan(phi_view)))
        denom = 2.0 * np.sin(theta_view) ** 2 * (
            delp * np.cos(psi_obs) *
            (np.cos(psi_obs) + secth * cotph * np.sin(psi_obs)) - 1.0)

        if np.max(np.abs(denom)) < 1.0e-6:
            print("triax_tpp2pqu: denominator ~ 0 -> invalid geometry.")
            return np.nan, np.nan, np.nan

        # These are temporary values of the squared intrinsic axial
        # ratios p^2 and q^2
        qintr = 1.0 - nom1minq2 / denom
        pintr = qintr + nomp2minq2 / denom

        # Quick check to see if we are not going to take the sqrt of
        # a negative number.
        if (np.min(qintr) < 1.0e-6) or (np.min(pintr) <= 1.0e-6):
            print(
                "triax_tpp2pqu: negative or too small intrinsic axis ratio squared "
                f"(min(q^2)={np.min(qintr)}, min(p^2)={np.min(pintr)})."
            )
            return np.nan, np.nan, np.nan

        # intrinsic axial ratios p and q
        qintr = np.sqrt(qintr)
        pintr = np.sqrt(pintr)

        # triaxiality parameter T = (1-p^2)/(1-q^2)
        triaxpar = (1.0 - pintr ** 2) / (1.0 - qintr ** 2)
        if (np.max(triaxpar) > 1.0) or (np.min(triaxpar) < 0.0):
            print(
                "triax_tpp2pqu: triaxiality parameter T out of [0, 1], "
                f"min(T)={np.min(triaxpar)}, max(T)={np.max(triaxpar)}.")
            return np.nan, np.nan, np.nan

        if np.max(qintr - pintr) > 0:
            print(
                "triax_tpp2pqu: intrinsic axis ordering violated (q > p). "
                f"max(q-p)={np.max(qintr - pintr)}.")
            return np.nan, np.nan, np.nan

        if np.min(qintr) <= 0.0:
            print(
                f"triax_tpp2pqu: intrinsic minor axis ratio q <= 0, min(q)={np.min(qintr)}."
            )
            return np.nan, np.nan, np.nan

        uintr = 1. / (np.sqrt(qobs_pot / np.sqrt(
            (pintr * np.cos(theta_view))**2 + (qintr * np.sin(theta_view))**2 *
            ((pintr * np.cos(phi_view))**2 + np.sin(phi_view)**2))))

        return pintr, qintr, uintr

    @staticmethod
    def acceleration(x, y, z,
                     viewing_angle,
                     ml,
                     surf_pot_pc,
                     sigobs_pot_pc,
                     qobs_pot,
                     psi_off=None,
                     epsrel=1e-4):
        """
        Gravitational acceleration of a triaxial MGE.

        Parameters
        ----------
        x, y, z : float or array-like
            Cartesian coordinates [pc].
        viewing_angle : tuple
            (theta, phi, psi) in degrees.
        surf_pot_pc, sigobs_pot_pc, qobs_pot : array-like
            Projected MGE quantities.
        ml : float
            Mass-to-light ratio.
        epsrel : float
            Relative accuracy for integration.

        Returns
        -------
        ax, ay, az : ndarray
            Acceleration components [(km/s)**2/pc].
        """

        # gravitational constant
        # G = 4.3009172706e-3   # [pc*(km/s)**2 / Msun]
        G = dyn.constants.GRAV_CONST_KM / dyn.constants.PARSEC_KM

        theta = viewing_angle[0]
        phi   = viewing_angle[1]
        psi   = viewing_angle[2]

        if psi_off is None:
            psi_off = np.zeros_like(qobs_pot)

        # deprojection: viewing angles -> intrinsic axis ratios
        pintr, qintr, uintr = TriaxialVisibleComponent.triax_tpp2pqu(
            theta,phi,psi,qobs_pot,psi_off)

        p_pot = pintr
        q_pot = qintr
        sig_pot_pc = sigobs_pot_pc
        sigintr_pc = sig_pot_pc / uintr

        V0 = (surf_pot_pc * (2.0 * np.pi * sig_pot_pc**2 * qobs_pot)
              * np.sqrt(2.0 / np.pi) / sigintr_pc**3 * G * ml)

        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        z = np.atleast_1d(np.asarray(z, dtype=float))

        ax = np.zeros_like(x)
        ay = np.zeros_like(x)
        az = np.zeros_like(x)

        # --- integrands ---
        def _acc_x_integrand(t, x_, y_, z_):
            dt = 1.0 - (1.0 - p_pot**2) * t**2
            et = 1.0 - (1.0 - q_pot**2) * t**2
            m2 = x_**2 + y_**2 / dt + z_**2 / et
            ker = (-np.exp(-t**2 * m2 / (2.0 * sigintr_pc**2))
                / np.sqrt(dt * et) * t**2 * x_)
            return np.sum(V0 * ker)

        def _acc_y_integrand(t, x_, y_, z_):
            dt = 1.0 - (1.0 - p_pot**2) * t**2
            et = 1.0 - (1.0 - q_pot**2) * t**2
            m2 = x_**2 + y_**2 / dt + z_**2 / et
            ker = (-np.exp(-t**2 * m2 / (2.0 * sigintr_pc**2))
                / np.sqrt(dt**3 * et)* t**2 * y_)
            return np.sum(V0 * ker)

        def _acc_z_integrand(t, x_, y_, z_):
            dt = 1.0 - (1.0 - p_pot**2) * t**2
            et = 1.0 - (1.0 - q_pot**2) * t**2
            m2 = x_**2 + y_**2 / dt + z_**2 / et
            ker = (-np.exp(-t**2 * m2 / (2.0 * sigintr_pc**2))
                / np.sqrt(dt * et**3)* t**2 * z_)
            return np.sum(V0 * ker)

        # main loop
        for i in range(len(x)):
            ax[i], _ = integrate.quad(_acc_x_integrand, 0.0, 1.0,
                                      args=(x[i], y[i], z[i]),epsrel=epsrel,)

            ay[i], _ = integrate.quad(_acc_y_integrand, 0.0, 1.0,
                                      args=(x[i], y[i], z[i]),epsrel=epsrel,)

            az[i], _ = integrate.quad(_acc_z_integrand, 0.0, 1.0,
                                      args=(x[i], y[i], z[i]),epsrel=epsrel,)

        return ax, ay, az


class BarDiskComponent(TriaxialVisibleComponent):
    """Rotating triaxial component with a MGE projected density (i.e. a bar),
    with viewing angles specified.

    Note: all bar components are constrained to have the same omega.

    """
    def __init__(self,
                 mge_pot=None,
                 mge_lum=None,
                 disk_pot=None,
                 disk_lum=None,
                 **kwds):
        super().__init__(**kwds)
        self.logger = logging.getLogger(f'{__name__}.{__class__.__name__}')
        self.qobs = np.nan
        self.par = ['theta', 'psi', 'phi']

    def validate_parset(self, par):
        # Skip validation as we already know the angles
        return True


class DarkComponent(Component):
    """Any dark component of the sytem, with no observed MGE or kinemtics

    This is an abstract layer and none of the attributes/methods are currently
    used.

    """
    def __init__(self,
                 density=None,
                 **kwds):
        # these have no observed properties (MGE/kinematics/populations)
        # instead they are initialised with an input density function
        self.density = density
        # self.mge = 'self.fit_mge()'
        super().__init__(visible=False,
                         kinematic_data=[],
                         population_data=[],
                         **kwds)

    def fit_mge(self,
                density,
                parameters,
                xyz_grid=[]):
        # fit an MGE for a given set of parameters
        # will be used in potential calculation
        rho = self.density.evaluate(xyz_grid, parameters)
        # self.mge = MGES.intrinsic_MGE_from_xyz_grid(xyz_grid, rho)

    def get_dh_legacy_strings(self, parset):
        """
        Generates and returns two strings needed for the legacy Fortran files.

        This method only applies to dark halo components.

        Parameters
        ----------
        parset : astropy table row
            Holds the parameter set.

        Returns
        -------
        specs : str
            A string with the legacy code and the number of parameters, space
            separated.
        par_vals : str
            The parameter values in the sequence legacy Fortran expects them,
            space separated.

        """
        try:
            legacy_code = self.legacy_code
            specs = f'{legacy_code} {len(self.parameters)}'
            par_vals = ''
            for par in self.par_names:
                p = f'{par}-{self.name}'
                par_vals += f'{parset[p]} '
            par_vals = par_vals[:-1]
            self.logger.debug(f'DH {self.__class__.__name__} legacy strings: '
                              f'{specs} / {par_vals}.')
            return specs, par_vals
        except AttributeError: # Only dh has a legacy code, Plummer: do nothing
            pass


class Plummer(DarkComponent):
    """A Plummer sphere

    Defined with parameters: M [mass, Msol] and a [scale length, arcsec]

    """
    def __init__(self, **kwds):
        super().__init__(symmetry='spherical', **kwds)

    def validate(self):
        par = ['m', 'a']
        super().validate(par=par)

    def density(x, y, z, pars):
        M, a = pars
        r = (x**2 + y**2 + z**2)**0.5
        rho = 3*M/4/np.pi/a**3 * (1. + (r/a)**2)**-2.5
        return rho

    def mass_enclosed(x, y, z, pars):
        M, a = pars
        r = (x**2 + y**2 + z**2)**0.5
        Menc = M*r**3/a**3*(1 + r**2/a**2)**(-1.5)
        return Menc

    @staticmethod
    def acceleration(x, y, z, par):
        """
        Gravitational acceleration of a Plummer sphere.

        Parameters
        ----------
        x, y, z : float or array-like
            Cartesian coordinates [pc]
        pars : tuple
            (M, a_pc [pc])

        Returns
        -------
        ax, ay, az : ndarray
            Acceleration components [(km/s)**2 / pc]
        """
        M    = par['m']
        a_pc = par['a_pc']

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)

        r2 = x**2 + y**2 + z**2
        denom = (r2 + a_pc**2)**1.5

        # G = 4.3009172706e-3  # pc * (km/s)**2 / Msun
        G = dyn.constants.GRAV_CONST_KM / dyn.constants.PARSEC_KM
        factor = -G * M / denom

        ax = factor * x
        ay = factor * y
        az = factor * z
        return ax, ay, az


class NFW(DarkComponent):
    """An NFW halo

    Defined with parameters: c [concentration, R200/scale] and f
    [dm-fraction, M200/total-stellar-mass]

    """
    par_names = ['c', 'f'] # parameter names in legacy sequence

    def __init__(self, **kwds):
        self.legacy_code = 1
        super().__init__(symmetry='spherical', **kwds)

    def validate(self):
        super().validate(par=self.par_names)


class NFW_m200_c(DarkComponent):
    """An NFW halo with the z=0 m200-c relation from Dutton & Maccio 14

    The relation: log10(c200) = 0.905 - 0.101 * log10(M200/(1e12/h)).
    Component defined with parameter f [dm-fraction, M200/total-stellar-mass]

    """
    par_names = ['f'] # parameter names in legacy sequence

    def __init__(self, **kwds):
        self.legacy_code = 1
        super().__init__(symmetry='spherical', **kwds)

    def validate(self):
        super().validate(par=self.par_names)

    def get_c200(self, system, parset):
        """
        Calculates and returns c200 (see Dutton & Maccio 2014).

        Parameters
        ----------
        system : a ``dyn.physical_system.System`` object
        parset : astropy table row
            Must contain dark matter fraction f-{self.name} and ml

        Returns
        -------
        float
            c200

        """
        stars = system.get_component_from_class(TriaxialVisibleComponent)
        M_stars_tot = stars.get_M_stars_tot(system.distMPc, parset)
        f = parset[f'f-{self.name}']
        h = system.H / 100
        #total mass of dark matter
        MvDM = f * M_stars_tot
        #dutton&maccio2014 (https://arxiv.org/pdf/1402.7073.pdf) Eq. (8)
        lc200 = 0.905 - 0.101*np.log10( MvDM/(1e12/h))

        return 10.**lc200

    def get_dh_legacy_strings(self, parset, system):
        """
        Generates and returns two strings needed for the legacy Fortran files.

        This method overrides the parent class' method because for legacy
        Fortran purposes, NFW_m200_c has two parameters. Note that NFW_m200_c
        needs an addiional parameter ``system``.

        Parameters
        ----------
        parset : astropy table row
            Holds the parameter set.

        Returns
        -------
        specs : str
            A string with the legacy code and the number of parameters, space
            separated.
        par_vals : str
            The parameter values in the sequence legacy Fortran expects them,
            space separated.

        """
        specs, par_vals = super().get_dh_legacy_strings(parset)
        c200 = self.get_c200(system, parset)
        specs = f'{self.legacy_code} 2'
        par_vals = f'{c200} {par_vals}'
        self.logger.debug(f'DH {self.__class__.__name__} legacy strings '
                          f'amended to {specs} / {par_vals}.')
        return specs, par_vals


    # c is concentration, f is dark mass fraction
    ## fixme: should derive rhocrit from (c,f) (?)
    rhocrit = 1
    def rhoc(c,f):
        return 200/3 * rhocrit * c**3 / (log(1 + c) - c/(1+c))
    def rc(c,f):
        return (3*M200(c,f)/(800*pi*rhocrit*c**3))**(1/3)
    def M200(c,f):
        return 800*pi/3*rhocrit*(rc*c)**3

    def potential(x, y, z, pars):
        c, f = pars
        d2 = x**2 + y**2 + z**2
        prefactor = 4*pi*G*rhoc(c,f)*(rc(c,f)**3)/sqrt(d2)
        if sqrt(d2)/rc >= 1:
            return prefactor * log(1 + sqrt(d2)/rc)
        else:
            return prefactor * 2 * atanh(sqrt(d2)/(2*rc(c,f) + sqrt(d2)))

    def density(x, y, z, pars):
        c, f = pars
        r = np.sqrt(x**2 + y**2 + z**2)
        rho = rc(c,f)**3*rhoc(c,f)/(r*(r+rc(c,f))**2)
        return rho

    def mass_enclosed(x, y, z, pars):
        c, f = pars
        r = np.sqrt(x**2 + y**2 + z**2)
        Menc = 4*np.pi*rc(c,f)**3*rhoc(c,f)*(np.log(1 + r/rc(c,f)) - (r/rc(c,f))/(1 + r/rc(c,f)))
        return Menc

class Hernquist(DarkComponent):
    """A Hernquist sphere

    Defined with parameters: rhoc [central density, Msun/km^3] and rc [scale
    length, km]

    """
    par_names = ['rhoc', 'rc'] # parameter names in legacy sequence

    def __init__(self, **kwds):
        self.legacy_code = 2
        super().__init__(symmetry='spherical', **kwds)

    def validate(self):
        super().validate(par=self.par_names)

    def potential(x, y, z, pars):
        rhoc, rc = pars
        r = np.sqrt(x**2 + y**2 + z**2)
        psi = 2*np.pi*G*rhoc*rc**2/(1 + r/rc)
        return psi

    def density(x, y, z, pars):
        rhoc, rc = pars
        r = np.sqrt(x**2 + y**2 + z**2)
        rho = rc**4*rhoc/(r*(r+rc)**3)
        return rho

    def mass_enclosed(x, y, z, pars):
        rhoc, rc = pars
        Menc = 2*np.pi*r**2*rc**3*rhoc/(r + rc)**2
        return Menc

class TriaxialCoredLogPotential(DarkComponent):
    """A TriaxialCoredLogPotential

    see e.g. Binney & Tremaine second edition p.171
    Defined with parameters: p [B/A], q [C/A], Rc [core radius, kpc], Vc
    [asympt. circular velovity, km/s]

    """
    par_names = ['Vc', 'Rc', 'p', 'q'] # parameter names in legacy sequence

    def __init__(self, **kwds):
        self.legacy_code = 3
        super().__init__(symmetry='triaxial', **kwds)

    def validate(self):
        super().validate(par=self.par_names)

    def potential(x, y, z, pars):
        rc, vc, p, q = pars
        m = x**2 + y**2/p**2 + z**2/q**2
        psi = -0.5*vc**2*np.log(rc**2 + m)
        return psi

    def density(x, y, z, pars):
        rc, vc, p, q = pars
        m = x**2 + y**2/p**2 + z**2/q**2
        rho = vc**2/(4*np.pi*G*(m+rc**2)**2)*( (m+rc**2)*(1 + 1/p**2 + 1/q**2) - 2*(x**2 + y**2/p**4 + z**2/q**4))
        return rho

    # this implementation assumes 1 > p > q
    def mass_enclosed(x, y, z, pars):
        rc, vc, p, q = pars
        r = np.sqrt(x**2 + y**2 + z**2)
        xx = r**2/(r**2/q**2 + rc**2)
        yy = r**2/(r**2/p**2 + rc**2)
        zz = r**2/(r**2 + rc**2)
        phi = np.arccos(np.sqrt(xx/zz))
        m = (zz-yy)/(zz-xx)
        Menc = r*vc**2/G * (1 - rc**2*r/np.sqrt((r**2+rc**2)*(r**2/p**2+rc**2)*(r**2/q**2+rc**2))*(zz-xx)**(-0.5)*special.ellipkinc(phi,m))


class GeneralisedNFW(DarkComponent):
    """A GeneralisedNFW halo

    from Zhao (1996)
    Defined with parameters: concentration [R200/NFW scale length], Mvir [Msol],
    inner_log_slope []

    """
    par_names = ['c', 'Mvir', 'gam'] # parameter names in legacy sequence

    def __init__(self, **kwds):
        self.legacy_code = 5
        super().__init__(symmetry='triaxial', **kwds)

    def validate(self):
        super().validate(par=self.par_names)

    def validate_parset(self, par):
        """
        Validates the GeneralisedNFW's parameters.

        Requires c and Mvir >0, and gam leq 1

        Parameters
        ----------
        par : dict
            { "p":val, ... } where "p" are the component's parameters and
            val are their respective values

        Returns
        -------
        bool
            True if the parameter set is valid, False otherwise

        """
        if (par['c']<0.) or (par['Mvir']<0.) or (par['gam']>1):
            is_valid = False
        else:
            is_valid = True
        return is_valid

    @staticmethod
    def density(x, y, z, halo_pars):
        '''
        Parameters
        ----------
        x, y, z : unit: pc
        pars = (rhoc, rc, gam)

        Returns
        -------
        rho : float
            Scale density, unit : Msun/pc**3
        '''
        rhoc, rc, gamma = halo_pars
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)

        r = np.sqrt(x**2 + y**2 + z**2)
        rho = rhoc * rc**3 * r**(-gamma) * (r + rc)**(gamma-3)
        return rho

    @staticmethod
    def mass_enclosed(x, y, z, halo_pars):
        '''
        Parameters
        ----------
        x, y, z : unit: pc
        pars = (rhoc, rc, gam)

        Returns
        -------
        Menc : float
            Mass, unit : Msun
        '''
        rhoc, rc, gamma = halo_pars

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)

        r = np.sqrt(x**2 + y**2 + z**2)

        xi = r/(r + rc)
        Menc = (4 * np.pi * rc**3 * rhoc * xi**(3-gamma)/(3-gamma)
                * special.hyp2f1(3-gamma,1,4-gamma,xi))
        return Menc

    @staticmethod
    def acceleration(x, y, z, par):
        """
        Gravitational acceleration of the gNFW halo.

        Parameters
        ----------
        x, y, z : float or array-like
            Cartesian coordinates [pc].
        par : dict
            Parameter dictionary from parset.

        Returns
        -------
        ax, ay, az : ndarray
            Acceleration components [(km/s)**2/pc].
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)

        r = np.sqrt(x**2 + y**2 + z**2)

        rhoc, rc, gamma = GeneralisedNFW.convert_parset(par)

        halo_pars = (rhoc, rc, gamma)
        M_enc = GeneralisedNFW.mass_enclosed(x, y, z, halo_pars)

        # G = 4.3009172706e-3                # [pc*(km/s)**2/Msun]
        G = dyn.constants.GRAV_CONST_KM / dyn.constants.PARSEC_KM
        factor = -G * M_enc / r**3

        ax = factor * x
        ay = factor * y
        az = factor * z

        return ax, ay, az


class StellarBlackHoles(DarkComponent):
    """A subcluster of stellar-mass black holes

    Spherical Zhao (1996) alpha-beta-gamma double power law::

        rho(r) = rho0 * (r/a)**-gamma * (1 + (r/a)**alpha)**(-(beta-gamma)/alpha)

    Config parameters: m [total sBH mass, Msun], a [scale radius, arcsec],
    alpha [transition sharpness], beta [outer log-slope], gamma [inner
    log-slope].

    The profile family was chosen by fitting both the PhaseFlow relaxed cusp
    and the GCfit/LIMEPY posterior jointly on rho(r) and M(<r); see
    ``dev_notes/sbh_profile_fits/`` and the design spec.

    Note ``beta > 3`` is required for the total mass to converge and
    ``gamma < 3`` for M(<r) to converge at the origin. ``gamma == 2``
    exactly is excluded because it makes the beta-function recurrence
    divide by zero; the physical content there is a logarithmic limit and
    the parameter is continuous.
    """
    # legacy sequence: rhoc replaces m, and a is in km not arcsec
    par_names = ['rhoc', 'a', 'alpha', 'beta', 'gamma']
    # config/sampled parameter names
    par = ['m', 'a', 'alpha', 'beta', 'gamma']

    def __init__(self, **kwds):
        self.legacy_code = 6
        super().__init__(symmetry='spherical', **kwds)
        self.logger = logging.getLogger(f'{__name__}.{__class__.__name__}')

    def validate(self):
        super().validate(par=self.par)

    def validate_parset(self, par):
        """
        Validate the sBH parameter values.

        Parameters
        ----------
        par : dict
            { "p":val, ... } where "p" are the component's parameters and
            val are their respective values

        Returns
        -------
        bool
            True if the parameter set is valid, False otherwise

        """
        ok = (par['m'] > 0.
              and par['a'] > 0.
              and par['alpha'] > 0.
              and par['beta'] > 3.
              and par['gamma'] < 3.
              and abs(par['gamma'] - 2.) > 1e-6)
        if not ok:
            self.logger.debug(f'Invalid sBH parset {dict(par)}: needs m>0, '
                              'a>0, alpha>0, beta>3, gamma<3, gamma!=2.')
        return bool(ok)

    @staticmethod
    def incomplete_beta(x, p, q):
        """Unregularised incomplete beta ``B(x; p, q)``, valid for q <= 0.

        ``B(x;p,q) = int_0^x u**(p-1) * (1-u)**(q-1) du``.

        For ``q > 0`` this is ``betainc(p,q,x) * beta(p,q)``. For ``q <= 0``
        the complete beta is undefined, so we step down from a positive-q
        evaluation using the contiguous relation::

            B(x;p,q) = [ (p+q) * B(x;p,q+1) - x**p * (1-x)**q ] / q

        This is the same recurrence the Fortran uses, and is why
        ``zh_betai`` is never called with a non-positive second argument.

        Parameters
        ----------
        x : float
            upper limit, 0 < x < 1
        p : float
            first parameter, must be > 0
        q : float
            second parameter, may be <= 0 but must not be a non-positive
            integer (0, -1, -2, ...): the downward recurrence used for
            q <= 0 steps through every ``qq = q, q+1, ..., 0`` on its way
            up, and hits a division by zero at ``qq == 0``.

        Returns
        -------
        float
            the incomplete beta function value

        Raises
        ------
        ValueError
            if q is a non-positive integer, where the recurrence used for
            q <= 0 divides by zero.

        """
        if q > 0.:
            return special.betainc(p, q, x) * special.beta(p, q)
        if q == np.floor(q):
            raise ValueError(
                f'incomplete_beta: q={q} is a non-positive integer; the '
                'q<=0 downward recurrence divides by zero at qq=0.')
        n = int(np.ceil(1. - q)) + 1
        val = special.betainc(p, q + n, x) * special.beta(p, q + n)
        for j in range(n, 0, -1):
            qq = q + j - 1.
            val = ((p + qq) * val - x ** p * (1. - x) ** qq) / qq
        return val

    @staticmethod
    def density(x, y, z, pars):
        '''
        Parameters
        ----------
        x, y, z : float or array-like
            Cartesian coordinates, same length units as ``a``
        pars : tuple
            (rho0, a, alpha, beta, gamma)

        Returns
        -------
        rho : float or ndarray
            density, in mass units of rho0
        '''
        rho0, a, alpha, beta, gamma = pars
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)
        r = np.sqrt(x**2 + y**2 + z**2)
        xx = r / a
        return rho0 * xx**(-gamma) * (1. + xx**alpha)**(-(beta-gamma)/alpha)

    @staticmethod
    def mass_enclosed(x, y, z, pars):
        '''
        Parameters
        ----------
        x, y, z : float or array-like
            Cartesian coordinates, same length units as ``a``
        pars : tuple
            (rho0, a, alpha, beta, gamma)

        Returns
        -------
        Menc : float or ndarray
            mass within r, = 4 pi a^3 rho0 / alpha * B(t; (3-g)/al, (b-3)/al)
            with t = (r/a)^alpha / (1 + (r/a)^alpha)
        '''
        rho0, a, alpha, beta, gamma = pars
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)
        r = np.sqrt(x**2 + y**2 + z**2)
        xx = r / a
        t = xx**alpha / (1. + xx**alpha)
        p = (3. - gamma) / alpha
        q = (beta - 3.) / alpha
        # both p and q are > 0 given gamma < 3 and beta > 3
        bi = special.betainc(p, q, t) * special.beta(p, q)
        return 4. * np.pi * a**3 * rho0 / alpha * bi

    @staticmethod
    def rho0_from_mass(m, a, alpha, beta, gamma):
        """Scale density giving a total mass ``m``.

        ``M_tot = 4 pi a^3 rho0 / alpha * B((3-gamma)/alpha, (beta-3)/alpha)``
        using the *complete* beta, which converges only for beta > 3.

        Parameters
        ----------
        m : float
            total sBH mass
        a : float
            scale radius
        alpha, beta, gamma : float
            shape exponents

        Returns
        -------
        float
            rho0, in mass units of m over length units of a cubed

        """
        b_complete = special.beta((3. - gamma) / alpha, (beta - 3.) / alpha)
        return m * alpha / (4. * np.pi * a**3 * b_complete)

    @staticmethod
    def _outer_tail(r, rho0, a, alpha, beta, gamma):
        """4 pi int_r^inf r' rho(r') dr', the potential's outer term.

        In terms of ``y = (r/a)**alpha`` the integral is

            (4 pi a^2 rho0 / alpha) * I(y),
            I(y) = int_y^inf s**(q-1) * (1+s)**-(p+q) ds

        with ``p = (beta-2)/alpha > 0`` and ``q = (2-gamma)/alpha``, which
        is <= 0 for the fitted gamma > 2 profiles. ``I`` is delegated to
        ``_outer_tail_integral``; see there for why this is parametrised
        by ``y`` and not by ``x = 1/(1+y)``.

        Parameters
        ----------
        r : float
            radius [pc], scalar
        rho0, a, alpha, beta, gamma : float
            profile parameters, see ``density``

        Returns
        -------
        float
        """
        y = (r / a) ** alpha
        p_out = (beta - 2.) / alpha
        q_out = (2. - gamma) / alpha
        val = StellarBlackHoles._outer_tail_integral(y, p_out, q_out)
        return 4. * np.pi * a**2 * rho0 / alpha * val

    # Tolerance and hard cap shared by the two series below. Both converge
    # geometrically, so the tolerance is what stops them and the cap only
    # ever fires on a caller outside the documented domain -- where it
    # raises rather than returning a silently truncated sum. It has to be
    # generous: the x-series' ratio is ``x = 1/(1+y_c)`` and the split
    # ``y_c`` shrinks like 1/(p+q) (see ``_outer_tail_integral``), so the
    # term count goes like 37*(p+q). Randomly sampling the whole legal
    # (alpha, beta, gamma) box with p, |q| <= 200 peaked at 1346 terms.
    _SERIES_TOL = 1e-16
    _SERIES_MAXIT = 100000

    @staticmethod
    def _expm1_over_z(z):
        """``(exp(z)-1)/z``, accurate as z -> 0, where it tends to 1.

        Fortran has no ``expm1`` intrinsic, so the Task 6 port needs the
        three-term Maclaurin branch below for small ``|z|`` and a plain
        ``(exp(z)-1)/z`` otherwise.
        """
        if abs(z) < 1e-5:
            return 1. + z * (0.5 + z * (1. / 6. + z / 24.))
        return np.expm1(z) / z

    @staticmethod
    def _beta_series_small_x(x, p, q):
        """``B(x; p, q)`` by the series about x = 0; needs 0 <= x < 1.

        Expanding ``(1-u)**(q-1)`` binomially inside
        ``int_0^x u**(p-1) (1-u)**(q-1) du`` and integrating term by term::

            B(x;p,q) = x**p * Sum_{k>=0} e_k x**k / (p+k),
            e_0 = 1,  e_k = e_{k-1} * (k-q) / k

        ``e_k`` is the generalised binomial coefficient ``C(q-1,k)(-1)**k``,
        built by a recurrence so that non-integer ``q`` -- the normal case,
        ``q = (2-gamma)/alpha`` -- costs nothing extra. ``p > 0`` is
        guaranteed by ``beta > 3``, so ``p+k`` never vanishes and no sign of
        ``q`` is special: unlike ``incomplete_beta``'s downward recurrence
        this is well defined for integer ``q <= 0`` too.

        Terms fall off like ``x**k``, so the term count goes like
        ``37/(1-x)``. Conditioning: for ``q <= 0`` every ``e_k`` is
        positive, so the sum has no cancellation at all; for ``q > 0``,
        ``Sum_k |e_k| x**k`` stays bounded only while ``x`` is not too
        close to 1, which is why the ``q > 0`` callers below keep their
        argument small rather than letting ``x -> 1``.

        Parameters
        ----------
        x : float
            upper limit, 0 <= x < 1
        p : float
            first parameter, must be > 0
        q : float
            second parameter, any sign

        Returns
        -------
        float
        """
        if x <= 0.:
            return 0.
        total = 1. / p
        coef = 1.
        for k in range(1, StellarBlackHoles._SERIES_MAXIT):
            coef *= (k - q) / k
            term = coef * x ** k / (p + k)
            total += term
            # |e_k| grows while k < |q|, so only test once the ratio
            # (k-q)/k * x is safely below 1; then the remaining tail is
            # bounded by |term| * x / (1-x)
            if k > abs(q) and abs(term) * x <= (1. - x) \
                    * StellarBlackHoles._SERIES_TOL * abs(total):
                break
        else:
            raise RuntimeError(
                f'_beta_series_small_x: no convergence in '
                f'{StellarBlackHoles._SERIES_MAXIT} terms at x={x}, '
                f'p={p}, q={q}; x is too close to 1 for this series.')
        return x ** p * total

    @staticmethod
    def _outer_tail_integral(y, p, q):
        """``int_y^inf s**(q-1) (1+s)**-(p+q) ds``, for y > 0 and p > 0.

        This is the potential's outer term in the natural variable
        ``y = (r/a)**alpha``. Writing it instead as ``B(x; p, q)`` with
        ``x = 1/(1+y)`` is algebraically equivalent but numerically fatal
        for small ``r``: the information lives in ``1-x = y/(1+y)``, and
        once ``y`` drops below ~1e-16 a double ``x`` rounds to exactly 1,
        where the integral is not even finite for ``q <= 0``. Keeping
        ``y`` as the argument keeps the small quantity small rather than
        hiding it in the last bits of a number near 1.

        For ``q > 1`` the integral is comfortably finite at ``y = 0`` and
        is taken as ``B(p,q)`` minus an explicit shortfall; see the code.
        Everything below is the ``q <= 1`` path, which covers both the
        genuinely divergent ``q <= 0`` (gamma >= 2) and the near-divergent
        small positive ``q``. Two regimes, crossing over at ``y = y_c``:

        * ``y >= y_c``: ``x = 1/(1+y) <= 1/(1+y_c)``, away from the
          singular endpoint, so ``_beta_series_small_x(x, p, q)`` is used
          directly (the substitution ``u = 1/(1+s)`` turns the integral
          into exactly ``B(x;p,q)``).
        * ``y < y_c``: split at ``s = y_c``. The upper piece is the
          constant ``B(1/(1+y_c); p, q)``, again by the same series. On
          the lower piece ``s <= y_c``, so expanding ``(1+s)**-(p+q)``
          binomially converges like ``y_c**k``::

              int_y^y_c = Sum_k d_k [ y_c**(q+k) - y**(q+k) ] / (q+k),
              d_0 = 1,  d_k = -d_{k-1} * (p+q+k-1) / k

          The ``k = 0`` term is ``~ -y**q/q``, the divergent part for
          ``q < 0``, and it is evaluated directly from ``y`` rather than
          reconstructed from a cancellation -- which is the whole point.

        ``y_c = min(1/2, 1/(p+q))``. The bound on ``p+q = (beta-gamma)/
        alpha`` is what makes the second series unconditionally stable:
        ``Sum_k |d_k| y_c**k = (1-y_c)**-(p+q)``, which for
        ``y_c = 1/(p+q)`` tends to ``e`` while the sum itself tends to
        ``1/e`` -- under one digit of cancellation for any exponents. A
        fixed ``y_c = 1/2`` looks tempting (~90 terms) but blows up: at
        ``(alpha,beta,gamma) = (0.2,12,2.9)``, ``p+q = 45.5`` and the
        partial sums would peak far above the answer. The
        price of the adaptive split is that the *first* series then runs
        at ratio ``1 - 1/(p+q)`` and needs ~``37*(p+q)`` terms; that is
        cheap arithmetic and this is not a hot path.

        ``[ y_c**e - y**e ] / e`` with ``e = q+k`` is itself a
        cancellation when ``|e*L|``, ``L = ln(y/y_c)``, is small (``e``
        near zero, i.e. ``q`` near a non-positive integer). It is
        rewritten there as ``-y_c**e * L * (exp(e*L)-1)/(e*L)``, whose
        last factor is regular at 0. That also makes ``e == 0`` exactly
        give the correct ``-y_c**e * L`` limit instead of dividing by 0,
        so integer ``q <= 0`` (e.g. alpha=0.5, gamma=2.5) is handled --
        which the ``incomplete_beta`` recurrence cannot do.

        Parameters
        ----------
        y : float
            lower limit, > 0
        p : float
            must be > 0
        q : float
            any sign

        Returns
        -------
        float
        """
        if y <= 0.:
            raise ValueError(f'_outer_tail_integral: y={y} must be > 0.')
        if q > 1.:
            # Comfortably finite at y = 0: the shortfall from the limit
            # B(p,q) goes like y**q / q. Reach it by subtracting that
            # shortfall from the complete beta, taking the complement
            # exactly as y/(1+y) rather than through 1 - 1/(1+y).
            #
            # The threshold is 1, not 0. For 0 < q <= 1 this route loses
            # digits twice over: B(p,q) and the shortfall both blow up
            # like 1/q as q -> 0 (i.e. gamma -> 2) and cancel, and for
            # small q the shortfall is a large fraction of the answer
            # anyway -- 18% at y = 3e-7, q = 0.12. The series route below
            # has no such problem: its e_k are all positive whenever
            # q <= 1, so it sums without any cancellation at all.
            w = y / (1. + y)
            if w <= min(0.5, 1. / p):
                return special.beta(p, q) \
                    - StellarBlackHoles._beta_series_small_x(w, q, p)
            # w is bounded away from 0, so 1-x carries full precision and
            # the ordinary betainc (Fortran: zh_betai) is well conditioned
            return StellarBlackHoles.incomplete_beta(1. / (1. + y), p, q)
        c = p + q
        y_c = min(0.5, 1. / c)
        if y >= y_c:
            return StellarBlackHoles._beta_series_small_x(1. / (1. + y), p, q)
        log_ratio = np.log(y / y_c)
        total = StellarBlackHoles._beta_series_small_x(1. / (1. + y_c), p, q)
        coef = 1.
        for k in range(0, StellarBlackHoles._SERIES_MAXIT):
            if k > 0:
                coef *= -(c + k - 1.) / k
            e = q + k
            z = e * log_ratio
            yc_e = y_c ** e
            if abs(z) < 1.:
                # regular form: no cancellation, and safe at e == 0
                term = -coef * yc_e * log_ratio \
                    * StellarBlackHoles._expm1_over_z(z)
            else:
                term = coef * (yc_e - y ** e) / e
            total += term
            # successive |d_k| grow while k < c, so only start testing
            # once the ratio (c+k)/(k+1) * y_c is safely below 1
            if k > c and abs(term) * y_c <= (1. - y_c) \
                    * StellarBlackHoles._SERIES_TOL * abs(total):
                break
        else:
            raise RuntimeError(
                f'_outer_tail_integral: no convergence in '
                f'{StellarBlackHoles._SERIES_MAXIT} terms at y={y}, '
                f'p={p}, q={q}.')
        return total

    @staticmethod
    def potential(x, y, z, pars):
        '''
        Gravitational potential Phi (negative, and -> 0 at large r for
        gamma < 2; for gamma >= 2 it diverges as r -> 0, which is physical).

        Phi(r) = -G [ M(<r)/r + 4 pi int_r^inf r' rho dr' ]

        The outer term is
        ``(4 pi a^2 rho0 / alpha) * B(1-t; (beta-2)/alpha, (2-gamma)/alpha)``,
        whose second beta parameter is <= 0 when gamma >= 2. It is
        evaluated by ``_outer_tail`` in the variable ``y = (r/a)**alpha``
        rather than in ``x = 1-t``, which loses all precision as r -> 0;
        see ``_outer_tail_integral``.

        Parameters
        ----------
        x, y, z : float or array-like
            Cartesian coordinates [pc]
        pars : tuple
            (rho0, a, alpha, beta, gamma), rho0 in Msun/pc**3, a in pc

        Returns
        -------
        Phi : ndarray
            potential [(km/s)**2]
        '''
        rho0, a, alpha, beta, gamma = pars
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)
        r = np.sqrt(x**2 + y**2 + z**2)
        tail = np.vectorize(StellarBlackHoles._outer_tail)(
            r, rho0, a, alpha, beta, gamma)
        m_enc = StellarBlackHoles.mass_enclosed(x, y, z, pars)
        G = dyn.constants.GRAV_CONST_KM / dyn.constants.PARSEC_KM
        return -G * (m_enc / r + tail)

    @staticmethod
    def acceleration(x, y, z, par):
        """
        Gravitational acceleration of the sBH subcluster.

        Exact for all gamma < 3: ``a_r = -G M(<r) / r**2``.

        Parameters
        ----------
        x, y, z : float or array-like
            Cartesian coordinates [pc]
        par : dict
            must contain m [Msun], a_pc [pc], alpha, beta, gamma

        Returns
        -------
        ax, ay, az : ndarray
            Acceleration components [(km/s)**2/pc]
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)
        r = np.sqrt(x**2 + y**2 + z**2)

        # 'a_pc' only -- deliberately NO a_km fallback. Mixing the two
        # silently scales the profile by ~3e13. The km-unit path is the
        # legacy file, and get_dh_legacy_strings converts there separately.
        a_pc = par['a_pc']
        rho0 = StellarBlackHoles.rho0_from_mass(
            par['m'], a_pc, par['alpha'], par['beta'], par['gamma'])
        pars = (rho0, a_pc, par['alpha'], par['beta'], par['gamma'])
        m_enc = StellarBlackHoles.mass_enclosed(x, y, z, pars)

        G = dyn.constants.GRAV_CONST_KM / dyn.constants.PARSEC_KM
        factor = -G * m_enc / r**3
        return factor * x, factor * y, factor * z

    def get_dh_legacy_strings(self, parset, system):
        """
        Generate the two strings the legacy Fortran needs.

        Overrides the parent because the sampled parameters (m in Msun,
        a in arcsec) differ from the legacy sequence (rhoc in Msun/km**3,
        a in km). This mirrors ``NFW_m200_c``, which likewise injects a
        derived quantity, and ``Hernquist``, which likewise passes a scale
        density rather than a mass.

        Parameters
        ----------
        parset : astropy table row
            Holds the parameter set.
        system : a ``dyn.physical_system.System`` object
            Needed for the distance, to convert arcsec to km.

        Returns
        -------
        specs : str
            legacy code and number of parameters, space separated
        par_vals : str
            parameter values in the sequence legacy Fortran expects

        """
        m = parset[f'm-{self.name}']
        a_arcsec = parset[f'a-{self.name}']
        alpha = parset[f'alpha-{self.name}']
        beta = parset[f'beta-{self.name}']
        gamma = parset[f'gamma-{self.name}']
        a_km = a_arcsec * dyn.constants.ARC_KM(system.distMPc)
        rhoc = self.rho0_from_mass(m, a_km, alpha, beta, gamma)
        specs = f'{self.legacy_code} {len(self.par_names)}'
        par_vals = f'{rhoc} {a_km} {alpha} {beta} {gamma}'
        self.logger.debug(f'sBH legacy strings: {specs} / {par_vals} '
                          f'(from m={m} Msun, a={a_arcsec} arcsec)')
        return specs, par_vals


class StellarBlackHolesMGE(DarkComponent):
    """A fixed, externally-supplied sBH profile represented as an MGE.

    Filled in by Task 8. Carries an ``mge_pot`` whose Gaussians are
    concatenated into the potential MGE, so it needs no legacy code and no
    Fortran changes.
    """
    par_names = []

    def __init__(self, mge_pot=None, **kwds):
        self.mge_pot = mge_pot
        super().__init__(symmetry='spherical', **kwds)
        self.logger = logging.getLogger(f'{__name__}.{__class__.__name__}')


class Chi2Ext(Component):
    """External component independent of DYNAMITE orbit and weight calculations

    This component interfaces to an external class that implements a chi2
    calculation independent of DYNAMITE orbit integration and weight solving.
    That chi2 value is added to all three chi2 values right after weight
    solving and is used by the parameter generator.

    Parameters
    ----------
    ext_module : str
        the name of the module implementing the external :math:`\chi^2`
        calculation. The associated .py file should be in the Python path.
    ext_class : str
        the class name in the external module implementing the external
        :math:`\chi^2` calculation. It will be instantiated once, at the time
        the config file is read.
    ext_class_args : dict
        the class parameters, can be empty (``{}``)
    ext_chi2 : str
        the name of the ``ext_class`` method returning :math:`\chi^2`
        as a single ``float``. In DYNAMITE, it will be called right after
        weight solving, passing the entire current parset.
    """
    def __init__(self,
                 ext_module=None,
                 ext_class=None,
                 ext_class_args=None,
                 ext_chi2=None,
                 **kwds):
        super().__init__(**kwds)
        self.logger = logging.getLogger(f'{__name__}.{__class__.__name__}')
        if ext_module is None or ext_class is None or ext_class_args is None \
           or ext_chi2 is None:
            txt = 'ext_module, ext_class, ext_class_args, ext_chi2 ' \
                  'cannot be None.'
            self.logger.error(txt)
            raise ValueError(txt)
        self.contributes_to_potential = False
        self.visible = False
        self.logger.debug(f'Importing {ext_module=}')
        import importlib  # only used once and only if Chi2Ext component exists
        the_ext_module = importlib.import_module(ext_module)
        args = tuple(f'{a}={ext_class_args[a]}' for a in ext_class_args)
        self.logger.debug('Instantiating '
                          f'{ext_module}.{ext_class}({args}).')
        ext_object = getattr(the_ext_module, ext_class)(**ext_class_args)
        self.ext_chi2 = getattr(ext_object, ext_chi2)

    def validate(self):  # allow any parameter names
        pars = [self.get_parname(p.name) for p in self.parameters]
        super().validate(par=pars)

    def get_chi2(self, model_id, config):
        """
        Returns the chi2 value for the parameter set.

        Parameters
        ----------
        model_id : int
            Model ID in the all_models table.
        config : a ``dyn.config.DynamiteConfig`` object
            The current DYNAMITE configuration

        Returns
        -------
        float
            The chi2 value

        """
        self.logger.debug(f'Calling external chi2 method with {model_id=}.')
        return self.ext_chi2(model_id, config)

# end
