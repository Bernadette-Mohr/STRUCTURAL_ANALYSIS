import argparse
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import MDAnalysis as mda
from MDAnalysis.analysis.lineardensity import LinearDensity
import warnings
print(mda.__version__)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def remove_entries(z):
    for key in ['dim', 'slice_volume', 'charge_density', 'charge_density_stddev']:
        del z[key]
    return z


def calculate_linear_densities(universe, solute, lipid, lipid_name):
    sol_beads = list()
    lip_beads = list()
    solute_com_results = None
    if lipid_name == 'CDL2':
        solute_com = LinearDensity(select=solute, grouping='fragments', binsize=0.1)
        solute_com.run()
        solute_com_results = remove_entries(solute_com.results.z)
        # print(solute_com.results.z['mass_density'].shape)
        # print('solute')
        for atom in solute.atoms:
            bead = dict()
            bd = universe.select_atoms(f'name {atom.name}')
            sol_bead = LinearDensity(select=bd, grouping='atoms', binsize=0.1)
            sol_bead.run()
            sol_bead.results.z = remove_entries(sol_bead.results.z)
            bead[atom.name] = sol_bead.results.z
            sol_beads.append(bead)
        if len(sol_beads) < 5:
            diff = 5 - len(sol_beads)
            sol_beads.extend([None for d in range(diff)])

    # print('lipid')
    bead_name_list = list()
    for atom in lipid.atoms:
        bead = dict()
        if atom.name not in bead_name_list:
            bead_name_list.append(atom.name)
            # print(atom.name, atom.index)
            lp = universe.select_atoms(f'name {atom.name}')
            lip_bead = LinearDensity(select=lp, grouping='atoms', binsize=0.1)
            lip_bead.run()
            lip_bead.results.z = remove_entries(lip_bead.results.z)
            # print(lip_bead.results.z['mass_density'].shape)
            bead[atom.type] = lip_bead.results.z
            lip_beads.append(bead)

    return solute_com_results, sol_beads, lip_beads


class PrepareTrajectory:

    def __init__(self, coordinates, trajectory):
        self.coords = coordinates,
        self.traj = trajectory
        self.universe = mda.Universe(coordinates, trajectory, in_memory=True)

    def __del__(self):
        class_name = self.__class__.__name__
        # print(f'{class_name} deleted')

    def select_components(self):
        universe = self.universe

        solute = universe.select_atoms('resname MOL', updating=True)
        lipid = universe.select_atoms('((resname CDL2 and resid 2) and (name GL0 or name PO41 or name PO42 or name GL11 or '
                                      'name GL21 or name GL22 or name C1A1 or name C2A1 or name C1A2 or name C2A2)) or '
                                      '((resname POPG and resid 2) and (name GL0 or name PO4 or name GL1 or name GL2 or '
                                      'name C1A or name C1B))', updating=True)
        water = universe.select_atoms('(resname PW and name W) and around 11 resname MOL', updating=True)
        ions = universe.select_atoms('(resname PNA and name NAC) and around 11 resname MOL', updating=True)

        return universe, solute, lipid, water, ions


def load_trajectroy(cdl2_results, popg_results, dir_path, filename):
    solute_index = list()
    solute_columns = pd.MultiIndex.from_product([['SOLUTE'], ['COM', 'bead1', 'bead2', 'bead3', 'bead4', 'bead5']])
    sol_df = pd.DataFrame(columns=solute_columns)
    # print(solute_df)
    cl_columns = pd.MultiIndex.from_product([['CDL2'], ['GL0', 'PO41', 'PO42', 'GL11', 'GL21', 'GL22', 'C1A1', 'C2A1',
                                                         'C1A2', 'C2A2']])
    cl_df = pd.DataFrame(columns=cl_columns)
    # print(cl_df)
    pg_columns = pd.MultiIndex.from_product([['POPG'], ['GL0', 'PO4', 'GL1', 'GL2', 'C1A', 'C1B']])
    pg_df = pd.DataFrame(columns=pg_columns)
    # print(pg_df)

    for cl_round_mols, pg_round_mols in tqdm(zip(sorted(cdl2_results.glob('ROUND_*/molecule_*'), reverse=False),
                                                 sorted(popg_results.glob('ROUND_*/molecule_*'), reverse=False)),
                                             total=439):
        round_ = cl_round_mols.parts[-2]
        molecule = cl_round_mols.parts[-1]
        solute_index.append((round_, molecule))

        idx = len(sol_df.index)
        for coords, trajs, in zip([cl_round_mols / f'prod-{molecule}.tpr', pg_round_mols / f'prod-{molecule}.tpr'],
                                 [cl_round_mols / f'prod-{molecule}.xtc', pg_round_mols / f'prod-{molecule}.xtc']):
            lipid_name = coords.parts[-4]
            prep_traj = PrepareTrajectory(str(coords), str(trajs))
            universe, solute, lipid, water, ions = prep_traj.select_components()
            if lipid_name == 'CDL2':
                sol_com_dens, sol_beads, lip_beads = calculate_linear_densities(universe, solute, lipid, lipid_name)
                sol_df.loc[idx, ('SOLUTE', 'COM')] = sol_com_dens
                sol_df.loc[idx, [('SOLUTE', 'bead1'), ('SOLUTE', 'bead2'), ('SOLUTE', 'bead3'), ('SOLUTE', 'bead4'),
                                 ('SOLUTE', 'bead5')]] = sol_beads
                cl_df.loc[idx, [('CDL2', 'GL0'), ('CDL2', 'PO41'), ('CDL2', 'PO42'), ('CDL2', 'GL11'),
                                ('CDL2', 'GL21'), ('CDL2', 'GL22'), ('CDL2', 'C1A1'), ('CDL2', 'C2A1'),
                                ('CDL2', 'C1A2'), ('CDL2', 'C2A2')]] = lip_beads
            else:
                _, _, lip_beads = calculate_linear_densities(universe, solute, lipid, lipid_name)
                pg_df.loc[idx, [('POPG', 'GL0'), ('POPG', 'PO4'), ('POPG', 'GL1'), ('POPG', 'GL2'), ('POPG', 'C1A'),
                                ('POPG', 'C1B')]] = lip_beads

    index = pd.MultiIndex.from_tuples(solute_index, names=['round', 'solute'])
    df = pd.concat([sol_df, cl_df, pg_df], axis=1)
    df.index = index
    df.to_pickle(dir_path / f'{filename}.pickle')
    print(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Provide input for methods of class PreprocessPipeline.')
    parser.add_argument('-clp', type=Path, required=True, help='Path to the MD simulation results for CL.')
    parser.add_argument('-pgp', type=Path, required=True, help='Path to the MD simulation results for PG.')
    parser.add_argument('-dir', '--directory', type=Path, required=True, help='Directory to save output.')
    parser.add_argument('-f', '--filename', type=str, required=False, default='position_density_data',
                        help='Filename for results output file.')

    args = parser.parse_args()

    load_trajectroy(args.clp, args.pgp, args.directory, args.filename)
