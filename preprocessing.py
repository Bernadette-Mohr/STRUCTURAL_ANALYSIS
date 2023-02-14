# MDAnalysis tools
import MDAnalysis as mda


class PreprocessPipeline:
    # Extract particle names and coordinates of the solutes and all environment components within the
    # cutoff distance of the long-range interactions around the COM of the solute.
    def __init__(self, coordinates, trajectory):
        self.universe = mda.Universe(coordinates, trajectory, in_memory=True)

    def preprocess_sim_data(self) -> object:
        # Particles are selected according to the cut-off distance defined by the long-range interactions of the
        # simulation around the COM of a central solute.
        # Return:
        #   sel_atoms (list): for each trajectory frame, all particle names within the long-range interaction cutoff
        #                     distance around the center of mass of the solute.
        #   sel_positions (list): for each trajectory frame, the xyz coordinates of all particles within the long-range
        #                         interaction cutoff distance around the center of mass of the solute.
        universe = self.universe
        print('Extracting relevant particles...')
        sel = universe.select_atoms('(resname MOL or around 11 resname MOL) and '
                                    'not (name WP or name WM or name NAP or name NAM)', updating=True)
        # list of particles, list of particle positions per frame
        sel_positions, sel_atoms = list(), list()
        for timestep in universe.trajectory:
            sel_atoms.append(sel.atoms)
            sel_positions.append(sel.positions)

        return sel_atoms, sel_positions
