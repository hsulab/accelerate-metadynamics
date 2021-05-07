import numpy as np

from ase.md.md import MolecularDynamics
from .md_utils import force_temperature

"""
verlet algorithm with velocity scaling
"""


class VelocityScaling(MolecularDynamics):
    def __init__(
        self, 
        atoms, 
        timestep = None, 
        scale_interval = None, 
        temperature = None, # in Kelvin
        trajectory = None, 
        logfile = None,
        loginterval = 1, 
        dt = None, 
        append_trajectory=False
    ):
        # FloK: rename dt -> timestep and make sure nobody is affected
        if dt is not None:
            import warnings
            warnings.warn('dt variable is deprecated; please use timestep.',
                          DeprecationWarning)
            timestep = dt
        if timestep is None:
            raise TypeError('Missing timestep argument')

        self.scale_on = False
        if temperature and scale_interval:
            self.scale_on = True
            self.scale_counter = 0
            self.temperature = temperature
            self.scale_interval = scale_interval

        MolecularDynamics.__init__(self, atoms, timestep, trajectory, logfile,
                                   loginterval,
                                   append_trajectory=append_trajectory)

    def step(self, f=None):

        atoms = self.atoms

        if f is None:
            f = atoms.get_forces()

        p = atoms.get_momenta()
        p += 0.5 * self.dt * f
        masses = atoms.get_masses()[:, np.newaxis]
        r = atoms.get_positions()

        # if we have constraints then this will do the first part of the
        # RATTLE algorithm:
        atoms.set_positions(r + self.dt * p / masses)
        if atoms.constraints:
            p = (atoms.get_positions() - r) * masses / self.dt

        atoms.set_momenta(p + 0.5 * self.dt * f)

        if self.scale_on:
            self.scale_counter += 1
            if self.scale_counter % self.scale_interval == 0:
                force_temperature(atoms, self.temperature)
                self.scale_counter = 0

        # TODO: check whether really needs this part
        # We need to store the momenta on the atoms before calculating
        # the forces, as in a parallel Asap calculation atoms may
        # migrate during force calculations, and the momenta need to
        # migrate along with the atoms.
        #atoms.set_momenta(p, apply_constraint=False)

        #f = atoms.get_forces(md=True)

        ## Second part of RATTLE will be done here:
        #atoms.set_momenta(atoms.get_momenta() + 0.5 * self.dt * f)

        return 
