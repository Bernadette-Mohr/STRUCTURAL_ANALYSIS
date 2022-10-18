# general python modules
import numpy as np
import pandas as pd
import regex as re
# SLATM tools, analysis
import qml


def get_atom_features(itp, col):
    """
        Return dataframe with info about residue structure in itp file.
        Only works for first molecule entry in itp file.

        itp (str): path to .itp file.
        col (str): name of one of the columns to detect where to start aggregating data.
        """
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
    # Generating "charges" for each type (they function as unique identifiers)
    charges_list = set(mapping_dict.values())
    charges_dict = {k: i for i, k in enumerate(charges_list, start=1)}
    compounds = pd.concat([compounds, new_df], ignore_index=True)
    # Generating many-body types
    charges_per_compound = [[charges_dict[mapping_dict[bead]] for bead in compound] for compound in
                            compounds['types'].tolist()]
    mbtypes = qml.representations.get_slatm_mbtypes(charges_per_compound)
    mbtypes = [list(i) for i in mbtypes]
    results = {'mbtypes': mbtypes, 'charges': charges_dict, 'mapping': mapping_dict}
    return results, compounds


def pad_bead_number(means):

    N = 5
    pad_value = np.zeros(means[0].size)
    pad_size = N - len(means)

    if len(means) < N:
        return [np.nan_to_num(avg, copy=True, nan=0.0, posinf=None, neginf=None)
                for avg in [*means, *[pad_value] * pad_size]]
    else:
        return [np.nan_to_num(avg, copy=True, nan=0.0, posinf=None, neginf=None) for avg in means]


class GenerateRepresentation:

    def __init__(self, mapping_dict, charges_dict, mbtypes, visuals=False):
        self.mapping_dict = mapping_dict
        self.charges_dict = charges_dict
        self.mbtypes = mbtypes
        self.visuals = visuals

    def get_charges(self, sel):
        """
        Returns list with unique charges per bead.
        """
        charges = []

        for i, at in enumerate(sel):
            name = at.name
            if name == 'GL0' and at.resname == "CDL2":
                name = 'GL0C'

            elif name == 'GL0' and at.resname == "POPG":
                name = 'GL0P'

            type_at = self.mapping_dict.get(name, "?")
            charges.append(self.charges_dict.get(type_at, "?"))

        # if "?" in charges:
        #     print(sub_sel, '\n')

        return charges

    def make_representation(self, atoms, positions) -> list:
        """
        For each frame, generates slatm representation.
        :rtype: list
        """
        rep_frames = []
        for idx, (at_frame, pos_frame) in enumerate(zip(atoms, positions)):
            charges_arr = self.get_charges(at_frame)
            # rep_frame = qml.representations.generate_slatm(pos_frame, charges_arr, self.mbtypes,
            #                                                local=True, sigmas=[0.3, .2], dgrids=[.2, .2], rcut=8.,
            #                                                rpower=6)
            rep_frame = qml.representations.generate_slatm(pos_frame, charges_arr, self.mbtypes,
                                                           local=True, sigmas=[.3, .2], dgrids=[.1, .1], rcut=8.,
                                                           rpower=6)

            rep_frames.append(rep_frame)

        solute = atoms[0].select_atoms('resname MOL')
        means = []
        for bead in solute:
            vec = [rep_frames[i][bead.index] for i in range(len(rep_frames))]
            mean = np.mean(np.array(vec), axis=0)
            means.append(mean)

        means = pad_bead_number(means)

        return means
