#!/usr/bin/env python3

"""
version 0.0.9
features
    * calculators, quip, dftb, vasp, emt
    * delta/pure machine learning
    * md engines, two stage
    * prepared samples
    * custom retrain options, speed simulation
"""

import os
import shutil
import subprocess

import time

import warnings
import logging

import json
import pickle

import numpy as np
import numpy.ma as ma

from ase import Atoms
from ase import units
from ase.io import jsonio
from ase.io import read, write
from ase.io.extxyz import read_extxyz, write_extxyz
from ase.io.dmol import write_dmol_arc

# calculators
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.emt import EMT
from ase.calculators.vasp import Vasp, Vasp2
from .xemt import XEMT
from .xdftb import XDFTB

# MD engines
from ase.md.verlet import VelocityVerlet
from .nosehoover import NoseHoover # NH thermostat without chains
from .velscale import VelocityScaling
from .aseplumed import AsePlumed

"""
Run various dftb calculations
    |
    V
Train dftb+ml model and pure ML model
    |
    V
Estimate error of single structures
    |
    V
Check key properties

Parameters:
    all from inputs.py

Calculators:
    EMT
    REAXFF
    DFTB+
    VASP
    GAP_FIT
    QUIP
"""

class FireFly(object):

    def __init__(
            self, 
            atoms, 
            constraints, # fix atoms
            # three classes 
            calculators: dict, 
            # parameters for three calculators
            calcParams: dict, 
            #atomic_energies_dict: dict,
            gapfitCommand: str, 
            mdEngines: list, 
            mdParams: list,
            timestep: float,
            maxSteps: int, 
            saveInterval: int = 100, 
            restart: bool = False,
            # with defaults
            prepared_samples: list = [], 
            prepared_basic_samples: list = [], 
            gapfitXmlName: str = 'GAP.xml',
            minSamples: int = 1,
            sampleInterval: int = 25,
            retrain_criteria: dict = {
                'min_interval': 100, 'max_interval': 1000, 'new_samples': 5
            }, 
            # tolerances in standard variance
            tolerances: dict = { 
                'energy': 0.008, 'forces': 0.04, 
                'force_percent': 0.05, 
                'force_threhold': 0.1
            },
            learnStress: bool = False,
            logLevel = logging.INFO,
            **kwargs
        ):
        """"""
        # whether load pickle 
        self.restart = restart
        self.restart_directory = './restart_files'
        self.restart_potential = './restart_files/GAP.xml'
        self.restart_atoms = './restart_files/saved_atoms.xyz'
        self.restart_basic = './restart_files/saved_basic.xyz'
        self.restart_reference = './restart_files/saved_reference.xyz'
        self.restart_status = './restart_files/status.pkl'

        self.saveInterval = saveInterval

        # initialize logger both to stdout and file
        self.init_logger('log.txt', logLevel)

        if self.restart:
            # check file consistent
            self.logger.info('read restart files')
            for file_path in [
                    self.restart_potential, self.restart_atoms, 
                    self.restart_basic, self.restart_reference,
                    self.restart_status
                ]:
                if not os.path.exists(file_path):
                    raise ValueError('%s for restart not exist.' %file_path)
        else:
            if os.path.exists(self.restart_directory):
                shutil.rmtree(self.restart_directory)
            os.makedirs(self.restart_directory)
            self.logger.info('remove old files and create new restart directory')

        # - parameters will be overwritten when restart 
        self.atoms = atoms
        self.constraints = constraints

        self.timestep = timestep

        # - check engines
        nengines = len(mdEngines)
        if nengines != len(mdParams):
            raise ValueError('Number of engines and paramsets not equal.')
        self.mdEngines = mdEngines
        self.mdParams = mdParams

        self.mdsteps = []
        for idx, (mdEngine, mdParam) in enumerate(zip(mdEngines, mdParams)):
            # check steps
            cur_nstep = mdParam.pop('nsteps', None)
            if (idx == 0):
                if (cur_nstep is None) or (cur_nstep != 0):
                    warnings.warn(
                        'the nsteps of first engine must be zero', 
                        RuntimeWarning
                    )
                    cur_nstep = 0
                    last_nstep = 0
            else:
                if (idx == nengines - 1):
                    # last engine
                    if (cur_nstep is None) or (cur_nstep >= maxSteps):
                        raise ValueError(
                            'the nsteps of last engine must be larger than last'
                        )
                else:
                    if (cur_nstep is None) or (cur_nstep <= last_nstep):
                        raise ValueError('nstep is illegal for engine %d' %idx)
            self.mdsteps.append(cur_nstep)
            self.logger.info(
                'The %d engine start step have been set to %d', idx, cur_nstep
            )

        self.usePlumed = False

        # - quip directory
        self.quipDirPath = './quip-worker'
        self.tempTrainXyzPath = os.path.join(self.quipDirPath, 'train.xyz')
        self.tempPotXmlPath = os.path.join(self.quipDirPath, gapfitXmlName)
        self.testAseXyzPath = os.path.join(self.quipDirPath, 'ase_temp.xyz')
        self.testQuipXyzPath = os.path.join(self.quipDirPath, 'quip.xyz')

        # - parameters can be initialised both start and restart 
        # calculators, these are calculators with parameters
        self.learnStress = learnStress

        self.calculators = calculators
        self.calcParams = calcParams

        self.iniNumFrames = minSamples
        self.collectInterval = sampleInterval

        # last step to train potential
        self.retrain_criteria = retrain_criteria
        self.retrain_status = {'nsamples': -10000, 'last': -10000} 

        # offer few structures in advance
        self.prepared_samples = prepared_samples
        self.prepared_basic_samples = prepared_basic_samples

        # training
        self.gapfitCommand = gapfitCommand

        self.maxSteps = maxSteps

        self.mdTrajPath = './traj.xyz'

        # check tolerance
        self.tolerances = tolerances
        self.tolerances['energy'] = (tolerances['energy']**2)*len(self.atoms)
        self.tolerances['forces'] = (tolerances['forces']**2)

        return

    def run(self):
        """run on-the-fly free energy calculation..."""
        # check status
        if self.restart:
            self.restart_train()
        else:
            self.initialise_train()

        # enter main loop
        while self.counter < self.maxSteps:
            self.logger.info('===== MD Step %d =====', self.counter)
            step_start_time = time.time()

            # ===== calculation with current calculator =====
            # calculate the structure with the current structure
            self.logger.info('\n----- Basic Info -----')
            self.logger.info(
                "%s calculation %s..." %(
                    self.current_calculator,
                    self.atoms.calc.__class__.__name__
                )
            )
            first_calculator = self.current_calculator # first called calculator

            self.current_forces = None
            self.potential_energies = {
                'basic': np.nan, 'corrected': np.nan, 'reference': np.nan
            }
            self.free_energies = {
                'basic': np.nan, 'corrected': np.nan, 'reference': np.nan
            }

            # - first calculation and save positions!
            start_time = time.time()
            self.current_forces = self.atoms.get_forces()
            end_time = time.time()
            self.logger.info(
                '%s cost time: %.3f s' % ('current calculation', end_time-start_time)
            )

            # - other info
            self.update_energy_info(
                self.current_calculator,
                self.atoms.calc.results
            )

            currentTemperature = self.atoms.get_temperature()
            self.logger.info(
                "\nTemperature %8.4f (K)\n" %currentTemperature
            )

            # save MD trajectory in xyz
            self.atoms.info['step'] = self.counter
            self.atoms.info['calculator'] = self.atoms.calc.__class__.__name__
            write(self.mdTrajPath, self.atoms, append=True)

            # ===== online AL =====
            al_start_time = time.time()

            self.recent_trained = False
            self.generate_initial_potential()
            if first_calculator == 'corrected':
                # this means we called corrected potential, which shall gives
                # error predictions
                self.check_correction_variance()

            # print out energies
            self.log_energy_info()

            al_end_time = time.time()
            self.logger.debug(
                '%s cost time: %.3f s' % ('AL', al_end_time-al_start_time)
            )

            # ===== md =====
            self.logger.info('\n----- MD Engine -----')
            # TODO: support more than two engines
            for idx, nsteps in enumerate(self.mdsteps):
                if self.counter == nsteps:
                    self.update_mdengine(idx)
                    self.cur_md_idx = idx
                    break

            # external
            if self.usePlumed:
                self.current_forces = self.asePlumed.external_forces(
                    self.counter,
                    new_energy = self.potential_energies[self.current_calculator],
                    new_forces = self.current_forces
                )

            # finish md step
            self.md.step(f=self.current_forces)

            # save status before check variance, 
            # once call retraining, the calculator will be updated
            if self.counter % self.saveInterval == 0:
                self.save_train()

            # finish the step
            self.counter += 1

            step_end_time = time.time()
            self.logger.info(
                '\n%s cost time: %.3f s\n' % ('step', step_end_time-step_start_time)
            )

        self.finalize_train()

        return

    def initialise_train(self):
        """"""
        self.logger.info('\n- Initialisation -\n')

        # directories
        if os.path.exists(self.quipDirPath):
            shutil.rmtree(self.quipDirPath)
            os.makedirs(self.quipDirPath)
        else:
            os.makedirs(self.quipDirPath)

        # write few files
        with open(self.mdTrajPath, 'w') as writer:
            writer.write('')

        with open(self.tempTrainXyzPath, 'w') as writer:
            writer.write('')

        for calcName in self.calculators.keys():
            with open(calcName+'.xyz', 'w') as writer:
                writer.write('')
                
        # few initial variables
        self.counter = 0

        self.basicAtomFrames = [] # structures from basic calculator
        self.refAtomFrames = []

        # constraint
        self.atoms.set_constraint(self.constraints)

        # check whether delta or pure active learning and set initial calculator
        if self.calculators['basic'] is None or self.calcParams['basic'] is None:
            self.logger.info('Pure Machine Learning Potential')
            self.delta_learning = False
            if self.iniNumFrames != 1:
                self.logger.warning('minSamples must be 1')
                self.iniNumFrames = 1
            # pure ML cannot calculate at first step, only offer a dummy calc
            self.update_calculator(None)
        else:
            self.logger.info('Delta Machine Learning Potential')
            self.delta_learning = True
            self.update_calculator('basic')

        # - use prepared samples 
        if self.prepared_samples:
            if self.delta_learning:
                self.logger.info('Prepared samples for delta machine learning!')
                if self.prepared_basic_samples:
                    self.logger.info('use calculated basic samples!')
                    # TODO: check atoms consistent 
                    assert len(self.prepared_basic_samples) == len(self.prepared_samples)
                    self.basicAtomFrames.extend(self.prepared_basic_samples)
                else:
                    self.logger.info('calculate basic samples right now!')
                    self.basicAtomFrames.extend(
                        self.call_calculation(self.prepared_samples, 'basic')
                    )
                self.refAtomFrames.extend(self.prepared_samples)
            else:
                # only for pure ML 
                self.logger.info('Prepared samples for pure machine learning!')
                self.logger.info('total training frames %d', len(self.refAtomFrames))
                self.refAtomFrames.extend(self.prepared_samples)
                self.logger.info('add prepared samples')
                self.logger.info('total training frames %d', len(self.refAtomFrames))

        # md
        self.cur_md_idx = 0

        return

    def finalize_train(self):
        """"""
        if self.usePlumed:
            self.asePlumed.finalize()

        self.logger.info(
            '\nSave last Potential File...\n' 
        )

        shutil.copyfile(self.tempPotXmlPath, './GAP_FINAL.xml')

        self.logger.info(
            '\nFinish at %s\n', 
            time.asctime( time.localtime(time.time()) )
        )

        # transfer to arc
        atomFrames = read(self.mdTrajPath, ':')
        write_dmol_arc('traj_final.arc', atomFrames)

        #
        self.save_train()

        self.logger.info(
            'use files below to restart \n' +
            'GAP.xml \n' + 
            'saved_atoms.xyz \n' + 
            'saved_basic.xyz \n' + 
            'saved_reference.xyz \n' + 
            'status.pkl'
        )

        return

    def save_train(self):
        """"""
        current_status = dict(
            #atoms = self.atoms, 
            counter = self.counter, 
            current_calculator = self.current_calculator, 
            md_idx = self.cur_md_idx, 
            #mdengine = self.md, 
            #usePlumed = self.usePlumed, 
            delta_learning = self.delta_learning, 
            retrain_status = self.retrain_status,
            #basicAtomFrames = self.basicAtomFrames, # structures from basic calculator
            #refAtomFrames = self.refAtomFrames, 
        )

        # avoid weakref situation
        if self.recent_trained:
            write(self.restart_atoms, self.load_results(self.atoms, atom_props=[], 
                config_props=[]))
        else:
            write(self.restart_atoms, self.load_results(self.atoms))

        write(self.restart_basic, self.basicAtomFrames)
        write(self.restart_reference, self.refAtomFrames)

        #if self.usePlumed:
        #    current_status.update(plumedInput = self.plumedInput)

        with open(self.restart_status, 'wb') as writer:
            pickle.dump(current_status, writer)

        shutil.copyfile('./GAP.xml', self.restart_potential)

        self.logger.info('\nsave current simulation status\n')

        return

    def restart_train(self):
        """"""
        self.logger.info('\n- Restart -\n')

        # directories
        if os.path.exists(self.quipDirPath):
            shutil.rmtree(self.quipDirPath)
            os.makedirs(self.quipDirPath)
        else:
            os.makedirs(self.quipDirPath)

        # write few files
        with open(self.mdTrajPath, 'w') as writer:
            writer.write('')

        with open(self.tempTrainXyzPath, 'w') as writer:
            writer.write('')

        for calcName in self.calculators.keys():
            with open(calcName+'.xyz', 'w') as writer:
                writer.write('')

        # start restart
        with open(self.restart_status, 'rb') as reader:
            current_status = pickle.load(reader)

        self.logger.info('restart from %d', current_status['counter'])

        # - recover atoms
        self.atoms = read(self.restart_atoms)
        self.atoms.set_constraint(self.constraints)

        shutil.copyfile(self.restart_potential, './GAP.xml')
        self.update_calculator(current_status['current_calculator'])

        #self.md = current_status['mdengine']
        self.cur_md_idx = current_status['md_idx']
        self.update_mdengine(current_status['md_idx'], False)
        self.logger.info(
            'Use a newly initialised md engine %s', 
            self.md.__class__.__name__
        )

        # - recover training
        self.delta_learning = current_status['delta_learning']
        self.retrain_status = current_status['retrain_status']

        self.counter = current_status['counter']

        # deal with unpicklable items 
        self.basicAtomFrames = read(self.restart_basic, ':') # structures from basic calculator
        self.refAtomFrames = read(self.restart_reference, ':')
        shutil.copyfile(self.restart_basic, './basic.xyz')
        shutil.copyfile(self.restart_reference, './reference.xyz')

        self.logger.info('total training structures %d', len(self.basicAtomFrames))

        return

    def generate_initial_potential(self):
        """Generate initial machine learning potential"""
        # collect
        if self.counter <= (self.iniNumFrames-1)*self.collectInterval and \
            self.counter % self.collectInterval == 0:
            if self.delta_learning:
                if self.current_calculator == 'basic':
                    basic_atoms = self.load_results(self.atoms)
                else:
                    raise ValueError('For initial potential generation, basic is must.')
            else:
                # pure learning, simply add atoms
                basic_atoms = self.atoms.copy()
                basic_atoms.set_calculator(
                    SinglePointCalculator(
                        basic_atoms, 
                        forces=np.zeros((len(self.atoms),3))
                    )
                )

            basic_atoms.info['step'] = self.counter
            self.basicAtomFrames.extend([basic_atoms])

            write('basic.xyz', basic_atoms, append=True)

        # batch training for initial potential
        if self.counter == (self.iniNumFrames-1)*self.collectInterval:
            # train DFTB + ML
            self.refAtomFrames.extend(
                self.call_calculation(
                    self.basicAtomFrames[-self.iniNumFrames:], # for prepared 
                    'reference'
                )
            )

            self.update_forces_info()

            # call training
            self.train_potential(self.basicAtomFrames, self.refAtomFrames)

        return

    def check_correction_variance(self):
        """"""
        predictedVariances = {}
        predictedVariances['max_energy_variance'] = -1.
        predictedVariances['max_atomic_energy_variance'] = -1.
        predictedVariances['max_forces_variance'] = -1.
        predictedVariances['max_frac_forces_variance'] = -1.

        self.logger.info('\n----- Check GP Potential Variance -----')

        energies_variance = self.atoms.calc.results['local_gap_variance']
        forces_variance = self.atoms.calc.results['gap_variance_gradient']

        predictedVariances = {}
        predictedVariances['max_energy_variance'] = \
            np.max(np.fabs(energies_variance))
        predictedVariances['max_forces_variance'] = \
            np.max([np.sqrt(np.sum(fv**2)) for fv in forces_variance])
        predictedVariances['max_frac_forces_variance'] = \
            np.max(np.fabs(forces_variance.flatten()))

        # check force percentage
        threhold = self.tolerances['force_threhold'] # avoid too small forces, 0.1
        max_force_var_percent = np.max(
            np.fabs(
                forces_variance.flatten() / 
                # remove zero forces due to constraints, and also too small ones
                ma.masked_less(np.fabs(self.current_forces.flatten()), threhold)
            )
        )

        self.logger.info('Energy Variance %f vs. %f', \
            predictedVariances['max_energy_variance'], self.tolerances['energy'])
        self.logger.info('Max Forces Variance %f vs. %f', \
            predictedVariances['max_forces_variance'], self.tolerances['forces'])
        self.logger.info('Max Frac Forces Variance %f vs. %f', \
            predictedVariances['max_frac_forces_variance'], self.tolerances['forces'])
        self.logger.info(
            'Max Frac Forces Variance Percentage %f vs. %f', 
            max_force_var_percent, self.tolerances['force_percent']
        )

        # check maximum error
        n_untrained = len(self.refAtomFrames) - self.retrain_status['nsamples']

        if (
            (predictedVariances['max_energy_variance'] >= 
            self.tolerances['energy']) # total energy variance
            or 
            #(predictedVariances['max_forces_variance'] >= 
            #self.tolerances['forces'])
            #or
            (predictedVariances['max_frac_forces_variance'] >= 
            self.tolerances['forces'])
            or
            max_force_var_percent >= self.tolerances['force_percent'] # 5%
        ):
            # call reference and training
            if (self.counter - self.retrain_status['last']) < self.retrain_criteria['min_interval']:
                self.logger.info(
                    '*too close to last training %d, skip this one, and new %d*', 
                    self.retrain_status['last'], n_untrained
                )
            else:
                self.logger.info('----- Add a new structure to dataset -----')

                # reference
                basic_atoms = self.atoms.copy()
                basic_atoms.info['step'] = self.counter

                if self.delta_learning:
                    self.basicAtomFrames.extend(
                        self.call_calculation([basic_atoms], 'basic')
                    )
                else:
                    # create dummy results
                    basic_atoms.set_calculator(
                        SinglePointCalculator(
                            basic_atoms, 
                            forces=np.zeros((len(self.atoms),3))
                        )
                    )
                    self.basicAtomFrames.extend([basic_atoms])

                # for the reference
                self.refAtomFrames.extend(
                    self.call_calculation([basic_atoms], 'reference')
                )

                # save train after calling reference
                #self.save_train()

                self.update_forces_info()

                # for the training
                self.train_potential(self.basicAtomFrames, self.refAtomFrames)
        else:
            # no new samples, but there are accumulated untrained ones
            if (n_untrained > 0):
                if (
                    n_untrained > self.retrain_criteria['new_samples'] or 
                    (self.counter - self.retrain_status['last']) > self.retrain_criteria['max_interval']
                ):
                    self.logger.info(
                        '*too far from last training %d, and new %d*', 
                        self.retrain_status['last'], n_untrained
                    )
                    self.train_potential(self.basicAtomFrames, self.refAtomFrames)
            else:
                pass

        return 

    def load_results(
            self, 
            atoms, 
            atom_props = ['forces'], 
            config_props = ['energy', 'free_energy']
        ):
        """ load results into a duplicate atoms object

        """
        per_atom_properties = atom_props
        per_config_properties = config_props
        if self.learnStress:
            per_config_properties.append('stress')

        # read results
        calc_results = atoms.calc.results.copy()

        results={}
        for key in (per_atom_properties+per_config_properties):
            value = calc_results.pop(key, None)
            if value is None:
                raise ValueError('Not calculated yet?')
            if key in per_config_properties:
                results[key] = value
                # for stress
            if key in per_atom_properties:
                if (len(value.shape) >= 1 and value.shape[0] == len(atoms)):
                    results[key] = value

        # load results
        saved_atoms = atoms.copy()
        if results != {}:
            calculator = SinglePointCalculator(saved_atoms, **results)
            saved_atoms.calc = calculator

        # add virial for GAP
        if self.learnStress:
            saved_atoms.info['virial'] = -saved_atoms.get_stress(voigt=False) * \
                saved_atoms.get_volume()

        return saved_atoms

    def train_potential(self, basicAtomFrames=None, refAtomFrames=None):
        """ command options output

            if training errors, always using binding energies.

        """

        self.logger.info('\n----- Train GP Potential -----')
        self.logger.info(
            'Number of structures in dataset %d', len(self.basicAtomFrames)
        )

        # check if attached calculators are all single point to save time
        if basicAtomFrames:
            self.logger.debug('\nbasic')
            for idx, test_atoms in enumerate(basicAtomFrames):
                if test_atoms.calc.__class__ != SinglePointCalculator:
                    raise ValueError('why not single?')
                self.logger.debug('structure %d', idx)
                self.logger.debug(test_atoms.get_forces())

        if refAtomFrames:
            self.logger.debug('\nreference')
            for idx, test_atoms in enumerate(refAtomFrames):
                if test_atoms.calc.__class__ != SinglePointCalculator:
                    raise ValueError('why not single?')
                self.logger.debug('structure %d', idx)
                self.logger.debug(test_atoms.get_forces())

        # delta learning, generate errors
        if self.delta_learning:
            #atomic_energies = self.atomic_energies_dict['energy']
            #atomic_free_energies = self.atomic_energies_dict['free_energy']
            properties = ['energy', 'free_energy', 'forces']
            errorAtomFrames = []
            for basicAtoms, refAtoms in zip(basicAtomFrames, refAtomFrames):
                results = {} # error results
                errorAtoms = basicAtoms.copy()
                # TODO:
                for prop in properties:
                    trainResults = basicAtoms.calc.results[prop]
                    #if prop == 'energy':
                    #    for sym in basicAtoms.get_chemical_symbols():
                    #        trainResults -= atomic_energies[sym]
                    #elif prop == 'free_energy':
                    #    for sym in basicAtoms.get_chemical_symbols():
                    #        trainResults -= atomic_free_energies[sym]
                    #else:
                    #    pass
                    results[prop] = refAtoms.calc.results[prop] - trainResults
                calc = SinglePointCalculator(errorAtoms, **results)
                errorAtoms.set_calculator(calc)
                if self.learnStress:
                    errorAtoms.info['virial'] = (
                        refAtoms.info['virial'] - basicAtoms.info['virial']
                    )
                errorAtomFrames.append(errorAtoms)
            write_extxyz(self.tempTrainXyzPath, errorAtomFrames)
            self.logger.info('write error xyz for delta-training')
        else: # for pure ML
            write_extxyz(self.tempTrainXyzPath, refAtomFrames)
            self.logger.info('write pure xyz for pure-training')

        command = self.gapfitCommand
        command += " > train.out"

        self.run_command(command, self.quipDirPath)

        # copy new pot
        shutil.copyfile(self.tempPotXmlPath, './GAP.xml')

        # update current calculator to correction
        self.update_calculator('corrected')

        self.recent_trained = True
        self.retrain_status = {
            'nsamples': len(self.refAtomFrames), 'last': self.counter
        }

        return

    def call_calculation(self, atomFrames: list, calcRole: str):
        """ call Basic/Reference to calculate reference energy and forces
            for result-safety, copy current results to a single point calculator
            
            atomFrames: list
                a list of atoms objects

            calcRole: str
                'basic', 'corrected', 'reference'

            !Calc: type
            !    ase calculators object

            !kwargs: dict
            !    parameters for calculators

        """
        start_time = time.time()

        Calc = self.calculators[calcRole]

        calc = Calc(**self.calcParams[calcRole])

        self.logger.info(
            '\n*** calculate train structures with %s ***\n',
            calcRole + '-' + Calc.__name__
        )

        calculatedAtomFrames = []
        for idx, basic_atoms in enumerate(atomFrames):
            self.logger.info('calculate structure %d', idx)
            atoms = basic_atoms.copy()
            atoms.set_calculator(calc)
            atoms.get_forces() # calculate

            calculatedAtoms = self.load_results(atoms)
            calculatedAtoms.info['calculator'] = \
                calcRole+'-'+atoms.calc.__class__.__name__

            write(calcRole+'.xyz', calculatedAtoms, append=True)
            calculatedAtomFrames.append(calculatedAtoms)

        end_time = time.time()
        self.logger.info(
            '\n%s cost time: %.3f s\n' % ('calculation', end_time-start_time)
        )

        return calculatedAtomFrames

    def update_mdengine(self, idx, reset_atoms=True):
        """"""
        # get engine info
        mdEngine = self.mdEngines[idx]
        mdParam = self.mdParams[idx]

        # check md atoms
        init_atoms = mdParam.pop('init_atoms', None)
        if (init_atoms is not None) and reset_atoms:
            # TODO: check consistent with self.atoms
            self.logger.info('reset atoms\n')
            self.atoms.set_positions(init_atoms.get_positions())
            self.atoms.set_velocities(init_atoms.get_velocities())

        # check plumed
        if self.usePlumed:
            self.asePlumed.finalize()
            self.usePlumed = False

        plumedInput = mdParam.pop('plumedInput', None)
        if plumedInput:
            if os.path.exists(plumedInput):
                self.usePlumed = True
                self.asePlumed = AsePlumed(
                    self.atoms, self.timestep, 
                    in_file = plumedInput
                )
                self.plumedInput = plumedInput
                self.logger.info('Enable PLUMED for enhancde sampling...\n')

        # generate md instance
        if mdEngine == VelocityVerlet:
            self.logger.info(
                'Using Pure Verlet Algorithm, ' + 
                'call two get_forces() in one step too slow\n'
            )
        elif mdEngine == VelocityScaling:
            self.logger.info(
                'Using Velocity Scaling, ' + 
                'temperature will be scaled at a given interval\n'
            )
        elif mdEngine == NoseHoover:
            self.logger.info(
                'Using Nose-Hoover thermostat, ' + 
                'temperature logged is the one at the last step\n'
            )
        else:
            pass
            #raise ValueError('%d MD Engine %s not supported.' %(idx, mdEngine.__name__))

        md = mdEngine(
            atoms = self.atoms, 
            timestep = self.timestep,
            **mdParam
        )

        self.md = md
        self.logger.info('current MD engine - %s\n', self.md.__class__.__name__)

        return

    def update_calculator(self, calcRole=None):
        """ update current calculator to the chosen one

            calcRole: str
                'basic', ...
        """
        if calcRole is None:
            calc = SinglePointCalculator(
                self.atoms, 
                energy = 0.0, 
                free_energy = 0.0, 
                forces = np.zeros((len(self.atoms),3))
            )
        else:
            Calc = self.calculators[calcRole]
            calc = Calc(**self.calcParams[calcRole])

        self.atoms.set_calculator(calc)

        self.logger.info('set current calculator to %s', calc.__class__.__name__)

        self.current_calculator = calcRole

        return


    @staticmethod
    def run_command(command, workDirPath):
        proc = subprocess.Popen(command, shell=True, cwd=workDirPath)

        errorcode = proc.wait()
        if errorcode:
            path = os.path.abspath(workDirPath)
            msg = ('Failed with command "{}" failed in '
                   '{} with error code {}'.format(command, path, errorcode))

            raise ValueError(msg)

        return

    def init_logger(self, logFilePath='log.txt', logLevel=logging.INFO):
        # log
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logLevel)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        if self.restart:
            fh = logging.FileHandler(filename=logFilePath, mode='a')
        else:
            fh = logging.FileHandler(filename=logFilePath, mode='w')

        fh.setLevel(logLevel)
        #fh.setFormatter(formatter)

        ch = logging.StreamHandler()
        ch.setLevel(logLevel)
        #ch.setFormatter(formatter)

        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

        # begin!
        self.logger.info(
            '\nStart at %s\n', 
            time.asctime( time.localtime(time.time()) )
        )

        return

    def init_mdengines(self):
        pass

    def update_energy_info(self, calc_name, calc_results):
        """"""
        if calc_name == None:
            pass
        else:
            self.potential_energies[calc_name] = calc_results['energy']
            self.free_energies[calc_name] = calc_results['free_energy']

        return

    def log_energy_info(self):
        content = '\n'
        content += ('{:<16s}  '*4+'\n').format(
            'energies(eV)', 'basic', 'corrected', 'reference'
        )
        content += ('{:<16s}  '+'{:>16.4f}  '*3+'\n').format(
                'potential', *list(self.potential_energies.values())
        )
        content += ('{:<16s}  '+'{:>16.4f}  '*3+'\n').format(
                'free', *list(self.free_energies.values())
        )
        content += '\n'

        self.logger.info(content)

        return

    def update_forces_info(self):
        # make current forces as the DFT values
        self.logger.info('update forces to DFT ones')
        self.current_forces = self.refAtomFrames[-1].calc.results['forces']

        self.update_energy_info(
            'reference', 
            self.refAtomFrames[-1].calc.results
        )

        return

if __name__ == '__main__':
    pass
