# general functionalities
import numpy as np
import pandas as pd
import regex as re
import tqdm

# Quantum machine learning: SLATM
import qml


def get_atom_features(itp):
    # Extracts information about solute structure from GROMACS topology (itp) file.
    # Return:
    #   bead_types (dict): "round": screening round, "molecule": identifier of solute in screening round,
    #                      "types": list of bead types present in a solute.
    rgx = re.compile(r'(?<=\[atoms\]\n).*(?=\[bonds\])', re.DOTALL | re.MULTILINE)
    molecule = itp.stem
    round_ = itp.parts[-4]
    bead_types = {'round': round_, 'molecule': molecule}

    with open(itp, mode='rt') as itp_file:
        itp = itp_file.read()
        match = re.search(rgx, itp).group(0).split('\n')
        match = [line.strip(';') for line in match if line and not line.startswith(';')]
        bead_types['types'] = [line.split('\t')[-4] for line in match]

    return bead_types


def make_mbtypes(compounds):
    # Renames GROMACS coarse-grained particles to Martini bead names and small ('S') type beads to normal beads.
    # Generates a list of unique integer identifiers for all particles in the samples and a list of all combinatorially
    # possible 2- and 3-body interactions.
    # Return:
    #   results (dict): "mbtypes" (list) : all combinatorially possible interactions, "charges" (list): unique integer
    #                   identifiers for all particles, "mapping" (dict): key-value pairs of GROMACS-Martini bead names.
    #   compounds (pandas dataframe): all
    mapping_dict = {'GL0P': 'P4',
                    'PO4': 'Qa',
                    'GL1': 'Na',
                    'GL2': 'Na',
                    'C1A': 'C1',
                    'D2A': 'C3',
                    'C3A': 'C1',
                    'C4A': 'C1',
                    'C1B': 'C1',
                    'C2B': 'C1',
                    'C3B': 'C1',
                    'C4B': 'C1',
                    'GL0C': 'Nda',
                    'PO41': 'Qa',
                    'GL11': 'Na',
                    'GL21': 'Na',
                    'C1A1': 'C1',
                    'C2A1': 'C1',
                    'D3A1': 'C3',
                    'C4A1': 'C1',
                    'C5A1': 'C1',
                    'C1B1': 'C1',
                    'C2B1': 'C1',
                    'D3B1': 'C3',
                    'C4B1': 'C1',
                    'C5B1': 'C1',
                    'PO42': 'Qa',
                    'GL22': 'Na',
                    'C1A2': 'C1',
                    'C2A2': 'C1',
                    'D3A2': 'C3',
                    'C4A2': 'C1',
                    'C5A2': 'C1',
                    'C1B2': 'C1',
                    'C2B2': 'C1',
                    'D3B2': 'C3',
                    'C4B2': 'C1',
                    'C5B2': 'C1',
                    'NAC': 'PQd',
                    'W': 'POL',
                    'T1': 'T1',
                    'SQ0': 'Q0',
                    'ST1': 'T1',
                    'ST3': 'T3',
                    'ST4': 'T4',
                    'ST5': 'T5',
                    'ST2': 'T2',
                    'T4': 'T4',
                    'T3': 'T3',
                    'Q0': 'Q0',
                    'T2': 'T2',
                    'T5': 'T5'}
    PG_beads = {'round': 'ALL', 'molecule': 'POPG', 'types': ['GL0P', 'PO4', 'GL1', 'GL2', 'C1A', 'D2A', 
                                                              'C3A', 'C4A', 'C1B', 'C2B', 'C3B', 'C4B']}
    CL_beads = {'round': 'ALL', 'molecule': 'CDL2', 'types': ['GL0C', 'PO41', 'GL11', 'GL21', 'C1A1', 'C2A1', 
                                                              'D3A1', 'C4A1', 'C5A1', 'C1B1', 'C2B1', 'D3B1', 
                                                              'C4B1', 'C5B1', 'PO42', 'GL21', 'GL22', 'C1A2', 
                                                              'C2A2', 'D3A2', 'C4A2', 'C5A2', 'C1B2', 'C2B2', 
                                                              'D3B2', 'C4B2', 'C5B2']}
    NA_beads = {'round': 'ALL', 'molecule': 'SODIUM', 'types': ['NAC']}
    W_beads = {'round': 'ALL', 'molecule': 'WATER', 'types': ['W']}
    rows = [PG_beads, CL_beads, NA_beads, W_beads]
    new_df = pd.DataFrame.from_dict(rows, orient='columns')
    # Generating "charges" for each type (in the case of coarse-grained particles, unique integer identifiers)
    charges_list = set(mapping_dict.values())
    charges_dict = {k: i for i, k in enumerate(charges_list, start=1)}
    compounds = pd.concat([compounds, new_df], ignore_index=True)
    # Generating many-body types: all combinatorially possible unique 2- and 3-body interactions
    charges_per_compound = [[charges_dict[mapping_dict[bead]] for bead in compound] for compound in
                            compounds['types'].tolist()]
    mbtypes = qml.representations.get_slatm_mbtypes(charges_per_compound)
    mbtypes = [list(i) for i in mbtypes]
    results = {'mbtypes': mbtypes, 'charges': charges_dict, 'mapping': mapping_dict}

    return results, compounds


def pad_bead_number(means):
    # If a solute consists of less than five coarse-grained beads, add an empty numpy vector of appropriate length
    # filled with zeros for each of the 5 - N missing beads. Also replaces eventual NaN values in averaged SLATM
    # representation by 0.0. Allows easy insertion in pandas dataframe.
    # Return:
    #   list: list containing SLATM representations and arrays filled with zeros as placeholders if the number of beads
    #         in a solute is less than five.

    N = 5
    pad_value = np.zeros(means[0].size)
    pad_size = N - len(means)

    if len(means) < N:
        return [np.nan_to_num(avg, copy=True, nan=0.0, posinf=None, neginf=None)
                for avg in [*means, *[pad_value] * pad_size]]
    else:
        return [np.nan_to_num(avg, copy=True, nan=0.0, posinf=None, neginf=None) for avg in means]


class GenerateRepresentation:
    # Handles generation of SLATM representations.
    def __init__(self, mapping_dict, charges_dict, mbtypes, visuals=False):
        self.mapping_dict = mapping_dict
        self.charges_dict = charges_dict
        self.mbtypes = mbtypes
        self.visuals = visuals

    def get_charges(self, sel):
        # Sets unique integer identifier for each bead.
        # Return:
        #   chargs (list): list of the integer identifiers in
        charges = []

        for i, at in enumerate(sel):
            name = at.name
            if name == 'GL0' and at.resname == "CDL2":
                name = 'GL0C'

            elif name == 'GL0' and at.resname == "POPG":
                name = 'GL0P'

            # Return all (wildcard "?") values of the mapping dict (Martini bead names).
            type_at = self.mapping_dict.get(name, "?")
            # Add the integers corresponding to the bead names to a list. Integers can appear multiple times in charges,
            # several GROMACS bead types map to the same Martini type in mapping_dict.
            charges.append(self.charges_dict.get(type_at, "?"))

        return charges

    def make_representation(self, atoms, positions) -> object:
        # For each bead in a solute and for each frame in the MD trajectory, generates a SLATM representation.
        # Averages over the representations of a bead to obtain one ensemble average SLATM representation.
        # Return:
        #   means (list): ensemble averaged SLATM representations for each bead in the solute, zero-arrays to increase
        #                 the number of representations to five if neccessary.
        rep_frames = []
        for idx, (at_frame, pos_frame) in enumerate(tqdm.tqdm(zip(atoms, positions), total=len(atoms), desc='Frames',
                                                              leave=True)):
            charges_arr = self.get_charges(at_frame)
            rep_frame = qml.representations.generate_slatm(pos_frame, charges_arr, self.mbtypes,
                                                           local=True, sigmas=[0.3, .2], dgrids=[.2, .2], rcut=8.,
                                                           rpower=6)
            rep_frames.append(rep_frame)

        print('Calculate configurational average...')
        solute = atoms[0].select_atoms('resname MOL')
        means = []
        for bead in solute:
            vec = [rep_frames[i][bead.index] for i in range(len(rep_frames))]
            mean = np.mean(np.array(vec), axis=0)
            means.append(mean)

        means = pad_bead_number(means)

        return means
