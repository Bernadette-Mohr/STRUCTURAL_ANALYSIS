import argparse
from pathlib import Path
import pandas as pd


def sum_hydrophobicity(types):
    dg_w_ol = {'Pqa': 25.535, 'Qda': -21.592, 'Pqd': 20.531, 'Q0': 20.479, 'Qd': 20.051, 'Qa': 20.051,
               'POL': 3.306, 'P4': 2.240, 'P3': 1.899, 'P5': 1.672, 'P2': 0.781, 'P1': 0.332,
               'Nda': -0.831, 'Nd': -0.831, 'Na': -0.831, 'N0': -1.440,
               'C5': -2.350, 'C4': -2.902, 'C3': -3.478, 'C2': -3.829, 'C1': -4.134,
               'T1': 2.052, 'T2': 1.906, 'T3': 0.098, 'T3a': 0.098, 'T3d': 0.098, 'T4': -2.455, 'T5': -3.129
               }
    hydrophobicity = 0.0
    for type_ in types:
        if type_.startswith('S'):
            type_ = type_[1:]
        hydrophobicity += dg_w_ol[type_]

    return hydrophobicity


def get_property_fractions(types):
    n_charges, n_hbonds, n_polars = 0, 0, 0

    for type_ in types:
        if type_.startswith('S'):
            type_ = type_[1:]

        if type_ == 'Q0':
            n_charges += 1
        elif type_ == 'T3':
            n_hbonds += 1
        elif type_ == 'T1' or type_ == 'T2':
            n_polars += 1
        else:
            pass

    n_charges /= len(types)
    n_hbonds /= len(types)
    n_polars /= len(types)

    return n_charges, n_hbonds, n_polars


def process_data(dir_path, bead_df, pc_df, filename):
    df = pd.read_pickle(pc_df)
    df[['round', 'solute']] = df.sol.str.split(expand=True)
    df.drop(['size', 'sol'], axis=1, inplace=True)
    df.rename(columns={0: 'PC1', 1: 'PC2', 2: 'PC3', 3: 'PC4', 4: 'PC5', 'dist': 'distance', 'selec': 'selectivity'},
              inplace=True)
    df = df[['round', 'solute', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'distance', 'selectivity']]

    bead_df = pd.read_pickle(bead_df)
    bead_df.rename(columns={'molecule': 'solute'}, inplace=True)
    bead_df['hydrophobicity'] = bead_df['types'].apply(sum_hydrophobicity)
    bead_df['charges'], bead_df['h_bonds'], bead_df['polarity'] = zip(*bead_df['types'].apply(get_property_fractions))

    df = pd.merge(df, bead_df[['round', 'solute', 'hydrophobicity', 'charges', 'h_bonds', 'polarity']],
                  on=['round', 'solute'], how='inner')

    df.to_pickle(dir_path / f'{filename}.pickle')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate various labels for solutes to identify principal components.')
    parser.add_argument('-dir', '--directory', type=Path, required=True, help='Directory for output dataframe.')
    parser.add_argument('-b', '--bead_types', type=Path, required=True, help='Dataframe with bead-types per solute')
    parser.add_argument('-pc', '--principal_components', type=Path, required=True,
                        help='Dataframe with principal components.')
    parser.add_argument('-fn', '--filename', type=str, required=False, default='PCA_solute_labels',
                        help='Name for output dataframe.')

    args = parser.parse_args()
    process_data(args.directory, args.bead_types, args.principal_components, args.filename)
