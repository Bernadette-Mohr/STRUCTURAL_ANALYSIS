import argparse
from pathlib import Path
import pandas as pd
import pickle


def sum_hydrophobicity(types, partCoeffs):
    # Calculates the average water-octanol partitioning per solute.
    dg_w_ol = pickle.load(open(partCoeffs, 'rb'))
    hydrophobicity = 0.0
    for type_ in types:
        if type_.startswith('S'):
            type_ = type_[1:]
        hydrophobicity += dg_w_ol[type_]
    hydrophobicity /= len(types)

    return hydrophobicity


def get_property_fractions(types):
    # Calculate ratio of specific bead types per solute
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


def process_data(dir_path, bead_df, pc_df, partCoeffs, filename):
    # Loads previously generated principal components, dataframe with a a list of bead types for each solute,
    # a dictionary with water-octanol partitioning coefficients per bead type, and generates descriptors.
    df = pd.read_pickle(pc_df)
    df[['round', 'solute']] = df.sol.str.split(expand=True)
    df.drop(['sol'], axis=1, inplace=True)
    if 'selec' in df.columns:
        df.rename(columns={0: 'PC1', 1: 'PC2', 2: 'PC3', 3: 'PC4', 4: 'PC5', 5: 'PC6', 'dist': 'distance',
                           'selec': 'selectivity'},
                  inplace=True)
        df = df[['round', 'solute', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'distance', 'selectivity']]
    else:
        df.rename(columns={0: 'PC1', 1: 'PC2', 2: 'PC3', 3: 'PC4', 4: 'PC5', 5: 'PC6', 'dist': 'distance'},
                  inplace=True)
        df = df[['round', 'solute', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'distance']]

    bead_df = pd.read_pickle(bead_df)
    bead_df.rename(columns={'molecule': 'solute'}, inplace=True)
    bead_df['hydrophobicity'] = bead_df['types'].apply(sum_hydrophobicity, args=(partCoeffs, ))
    bead_df['charges'], bead_df['h_bonds'], bead_df['polarity'] = zip(*bead_df['types'].apply(get_property_fractions))

    df = pd.merge(df, bead_df[['round', 'solute', 'hydrophobicity', 'charges', 'h_bonds', 'polarity']],
                  on=['round', 'solute'], how='inner')

    df.to_pickle(dir_path / f'{filename}.pickle')


if __name__ == '__main__':
    # Results, data and settings loaded from command line.
    parser = argparse.ArgumentParser('Generate various labels for solutes to identify principal components.')
    parser.add_argument('-dir', '--directory', type=Path, required=True, help='Directory for output dataframe.')
    parser.add_argument('-b', '--bead_types', type=Path, required=True, help='Dataframe with bead-types per solute')
    parser.add_argument('-pc', '--principal_components', type=Path, required=True,
                        help='Dataframe with principal components.')
    parser.add_argument('-dGp', '--partCoeffs', type=Path, required=False, default=Path('dG_w_ol.pkl'),
                        help='Path to pickled dictionary with bead-name: partitioning coefficient as key: value pairs, '
                             'containing water-octanol partitioning coefficients for CG beads in samples.')
    parser.add_argument('-fn', '--filename', type=str, required=False, default='PCA_solute_labels',
                        help='Name for output dataframe.')

    args = parser.parse_args()
    process_data(args.directory, args.bead_types, args.principal_components, args.partCoeffs, args.filename)
