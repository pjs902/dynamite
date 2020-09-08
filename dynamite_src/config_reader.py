#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:14:17 2020

@author: maindl
"""

# some tricks to add the current path to sys.path (so the imports below work)

import os.path
import sys

this_dir = os.path.dirname(__file__)
if not this_dir in sys.path:
    sys.path.append(this_dir)

# import required modules/packages

import yaml
import physical_system as physys
#from dynamite_src.physical_system import *
import parameter_space as parspace
import kinematics as kinem
import populations as popul
import mges as mge

class Settings(object):
    """
    Class that collects misc configuration settings
    """
    def __init__(self):
        self.orblib_settings = {}
        self.parameter_space_settings = {}

    def add(self, kind, values):
        if kind == 'orblib_settings':
            self.orblib_settings = values
        elif kind == 'parameter_space_settings':
            self.parameter_space_settings = values
        elif kind == 'legacy_settings':
            self.legacy_settings = values
        elif kind == 'io_settings':
            self.io_settings = values
        elif kind == 'weight_solver_settings':
            self.weight_solver_settings = values
        else:
            raise ValueError("""Config only takes orblib_settings
                             and parameter_space_settings
                             and io_settings
                             and weight_solver_settings""")

    def validate(self):
        if not(self.orblib_settings and self.parameter_space_settings and
               self.output_settings and self.weight_solver_settings):
            raise ValueError("""Config needs orblib_settings
                             and parameter_space_settings
                             and io_settings
                             and weight_solver_settings""")

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.__dict__})')


class ConfigurationReaderYaml(object):
    """
    Reads the configuration file and instantiates the objects
    self.system, ...
    """
    def __init__(self, filename=None, silent=False): # instantiate the objects here. instead of the dict, self.system will be a System object, etc.
        """
        Reads onfiguration file and instantiates objects. Does some rudimentary
        checks for consistency.

        Parameters
        ----------
        filename : string, needs to refer to an existing file including path
        silent : True suppresses output, default=False

        Raises
        ------
        FileNotFoundError
            If file does not exist or filename is None or not given.

        Returns
        -------
        None.

        """

        if filename is not None:
            with open(filename, 'r') as f:
                self.params = yaml.safe_load(f)
        else:
            raise FileNotFoundError('Please specify filename')

        self.system = physys.System() # instantiate System object
        self.settings = Settings() # instantiate Configuration object
#        self.__dict__ = par

        for key, value in self.params.items(): # walk through the file contents...

            # add components to system

            if key == 'system_components':
                if not silent:
                    print('model_components:')
                for comp, data_comp in value.items():
                    if not data_comp['include']:
                            if not silent:
                                print('', comp, '  ...ignored')
                            continue

                    # instantiate the component

                    if not silent:
                        print(f" {comp}... instantiating {data_comp['type']} object")
                    if 'contributes_to_potential' not in data_comp:
                        raise ValueError(f'Component {comp} needs contributes_to_potential attribute')
#                    c = globals()[data_comp['type']](contributes_to_potential = data_comp['contributes_to_potential'])
                    c = getattr(physys,data_comp['type'])(name = comp, contributes_to_potential = data_comp['contributes_to_potential'])

                    # initialize the componnt's paramaters, kinematics, and populations

                    par_list, kin_list, pop_list = [], [], []

                    # read parameters

                    if 'parameters' not in data_comp:
                        raise ValueError('Component ' + comp + ' needs parameters')
                    if not silent:
                        print(f" Has parameters {tuple(data_comp['parameters'].keys())}")
                    for par, data_par in data_comp['parameters'].items():
                        the_parameter = parspace.Parameter(name=par, **data_par)
                        par_list.append(the_parameter)
                    c.parameters = par_list

                    # read kinematics

                    if 'kinematics' in data_comp:   # shall we include a check here (e.g., only VisibleComponent can have kinematics?)
                        if not silent:
                            print(f" Has kinematics {tuple(data_comp['kinematics'].keys())}")
                        for kin, data_kin in data_comp['kinematics'].items():
                            # kinematics_set = kinem.Kinematics(name=kin, **data_kin)
                            kinematics_set = getattr(kinem,data_kin['type'])(name = kin, **data_kin)
                            kin_list.append(kinematics_set)
                        c.kinematic_data = kin_list

                    # read populations

                    if 'populations' in data_comp:   # shall we include a check here (e.g., only VisibleComponent can have populations?)
                        if not silent:
                            print(f" Has populations {tuple(data_comp['populations'].keys())}")
                        for pop, data_pop in data_comp['populations'].items():
                            populations_set = popul.Populations(name=pop, **data_pop)
                            pop_list.append(populations_set)
                        c.population_data = pop_list

                    if 'mge_file' in data_comp:
                        c.mge = mge.MGE(datafile=data_comp['mge_file'])

                    # add component to system
                    c.validate()
                    self.system.add_component(c)

            # add system parameters

            elif key == 'system_parameters':
                if not silent:
                    print('system_parameters...')
                    print(f' {tuple(value.keys())}')
                par_list = []
                for other, data in value.items():
                    par_list.append(parspace.Parameter(name=other, **data))
                setattr(self.system, 'parameters', par_list)
                    # if other == 'ml':
                    #     self.system.ml = parspace.Parameter(name=other, **data)
                    # else:
                    #     setattr(self.system, other, data)

            # add system attributes

            elif key == 'system_attributes':
                if not silent:
                    print('system_attributes...')
                    print(f' {tuple(value.keys())}')
                for other, data in value.items():
                    setattr(self.system, other, data)

            # add orbit library settings to config object

            elif key == 'orblib_settings':
                if not silent:
                    print('orblib_settings...')
                    print(f' {tuple(value.keys())}')
                self.settings.add('orblib_settings', value)

            # add parameter space settings to config object

            elif key == 'parameter_space_settings':
                if not silent:
                    print('parameter_space_settings...')
                    print(f' {tuple(value.keys())}')
                self.settings.add('parameter_space_settings', value)

            # add legacy settings to config object

            elif key == 'legacy_settings':
                if not silent:
                    print('legacy_settings...')
                    print(f' {tuple(value.keys())}')
                self.settings.add('legacy_settings', value)

            # add output settings to config object

            elif key == 'io_settings':
                print('hello')
                if not silent:
                    print('io_settings...')
                    print(f' {tuple(value.keys())}')
                self.settings.add('io_settings', value)

            # add weight_solver_settings to config object

            elif key == 'weight_solver_settings':
                if not silent:
                    print('weight_solver_settings...')
                    print(f' {tuple(value.keys())}')
                self.settings.add('weight_solver_settings', value)

            else:
                raise ValueError(f'Unknown configuration key: {key}')

        #self.system.validate()
        #self.config.validate()

        if not silent:
            print(f'**** System assembled:\n{self.system}')
            print(f'**** Settings:\n{self.settings}')

        self.validate()
        if not silent:
            print('**** Configuration validated')

        self.parspace = parspace.ParameterSpace(self.system)
        if not silent:
            print(f'**** Instantiated parameter space')
            print(f'**** Parameter space:\n{self.parspace}')

    def validate(self):
        """
        Validates the system and settings. This method is still VERY
        rudimentary and will be adjusted as we add new functionality to dynamite
        Currently, this method is geared towards legacy mode

        Returns
        -------
        None.

        """
        if sum(1 for i in self.system.cmp_list if isinstance(i, physys.Plummer)) != 1:
            raise ValueError('System needs to have exactly one Plummer object')
        if sum(1 for i in self.system.cmp_list if isinstance(i, physys.VisibleComponent)) != 1:
            raise ValueError('System needs to have exactly one VisibleComponent object')
        if sum(1 for i in self.system.cmp_list if issubclass(type(i), physys.DarkComponent) and not isinstance(i, physys.Plummer)) > 1:
            raise ValueError('System needs to have zero or one DM Halo object')
        if not ( 1 < len(self.system.cmp_list) < 4):
            raise ValueError('System needs to comprise exactly one Plummer, one VisibleComponent, and zero or one DM Halo object')

        for c in self.system.cmp_list:
            if issubclass(type(c), physys.VisibleComponent): # Check visible component
                if c.kinematic_data:
                    for kin_data in c.kinematic_data:
                        if kin_data.type != 'GaussHermite':
                            raise ValueError('VisibleComponent kinematics need GaussHermite type')
                else:
                    raise ValueError('VisibleComponent must have kinematics with type GaussHermite')
                if c.symmetry != 'triax':
                    raise ValueError('Legacy mode: VisibleComponent must be triaxial')
                continue
            if issubclass(type(c), physys.DarkComponent) and not isinstance(c, physys.Plummer): # Check allowed dm halos in legacy mode
                if type(c) not in [physys.NFW, physys.Hernquist, physys.TriaxialCoredLogPotential, physys.GeneralisedNFW]:
                    raise ValueError(f'DM Halo needs to be of type NFW, Hernquist, TriaxialCoredLogPotential, or GeneralisedNFW, not {type(c)}')

        if self.settings.parameter_space_settings["generator_type"] != 'GridSearch':
            raise ValueError('Legacy mode: parameter space generator_type must be GridSearch')


    # def read_parameters(self, par=None, items=None):
    #     """
    #     Will add each key-value pair in items to parameters object par by calling
    #     par.add(...) and subsequently calls the par.validate() method.

    #     Parameters
    #     ----------
    #     par : ...parameters object, optional
    #         The default is None.
    #     items : dictionary, optional
    #         The default is None.

    #     Returns
    #     -------
    #     None.

    #     """

    #     for p, v in items:
    #         par.add(name=p, **v)
    #     par.validate()