import numpy as np
import MDAnalysis as mda


class PreprocessPipeline:

    def __init__(self, coordinates, trajectory):
        self.universe = mda.Universe(coordinates, trajectory, in_memory=True)

    def __del__(self):
        class_name = self.__class__.__name__
        print(f'{class_name} deleted')

    def preprocess_sim_data(self) -> object:
        """
        Preprocesses all frames in simulation and returns TODO

        """
        universe = self.universe

        sel = universe.select_atoms('(resname MOL or around 11 resname MOL) and '
                                    'not (name WP or name WM or name NAP or name NAM)', updating=True)
        mol = universe.select_atoms('resname MOL', updating=True)

        # list of particle positions per frame
        sel_positions, mol_positions = list(), list()
        # list of atom selections per frame
        sel_atoms, mol_atoms = list(), list()
        for timestep in universe.trajectory:
            sel_atoms.append(sel.atoms)
            mol_atoms.append(mol.atoms)
            sel_positions.append(sel.positions)
            mol_positions.append(mol.positions)
        return sel_atoms, mol_atoms, sel_positions, mol_positions
