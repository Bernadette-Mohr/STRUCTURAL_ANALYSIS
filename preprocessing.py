import MDAnalysis as mda


class PreprocessPipeline:

    def __init__(self, coordinates, trajectory):
        self.universe = mda.Universe(coordinates, trajectory, in_memory=True)

    def preprocess_sim_data(self) -> object:
        """
        Preprocesses all frames in simulation and returns a list of selected particles and a list of their respective
         xyz coordinates. Particles are selected accoridng to the cut-off distance defined by the long-range
         interactions of the simulation around the COM of a central solute.
        """
        universe = self.universe
        print('Extracting relevant particles...')
        sel = universe.select_atoms('(resname MOL or around 11 resname MOL) and '
                                    'not (name WP or name WM or name NAP or name NAM)', updating=True)
        """
         list of partilces, list of particle positions per frame
        """
        sel_positions, sel_atoms = list(), list()
        for timestep in universe.trajectory:
            sel_atoms.append(sel.atoms)
            sel_positions.append(sel.positions)

        return sel_atoms, sel_positions
