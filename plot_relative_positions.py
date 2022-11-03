import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid', palette='deep')


def calculate_averages(sol):
    for stats in sol.loc[:, ('SOLUTE', 'COM')]:
        print(stats['mass_density'].shape)

    return 'foo', 'bar'


def plot_solute_com(lipids, sol):
    # print(sol.loc[:, ('SOLUTE', 'COM')])
    avg_pos, avg_stddev = calculate_averages(sol)


def generate_plots(dir_path, dataframe, plot_type, bead_type=None):
    if not dataframe.is_absolute():
        df_path = dir_path / dataframe
    else:
        df_path = dataframe
    df = pd.read_pickle(df_path)
    lipids = df.loc[:, ['CDL2', 'POPG']]

    if plot_type == 'com':
        print('Plotting average solute COM position.')
        sol = df.loc[:, [('SOLUTE', 'COM')]]
        plot_solute_com(lipids, sol)
    elif plot_type == 'type' and bead_type:
        pass
    else:
        pass

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Plot lipid headgroup vs solute positions.')
    parser.add_argument('-dir', '--directory', type=Path, required=True, help='Directory path for saving plots')
    parser.add_argument('-df', '--dataframe', type=Path, required=True, help='Pandas dataframe with position density '
                                                                             'data for system.')
    parser.add_argument('-p', '--plot', type=str, required=False, default='com', choices=['com', 'type'],
                        help='Keyword selecting if solute COM or individual beads should be plotted. OPtions are: '
                             '\'com\', \'type\'. If \'type\' is given, please pass the name of the solute bead type to '
                             'select.')
    parser.add_argument('-t', '--type', type=str, required=False, help='Name of the solute particle name whose position'
                                                                       ' is to be plotted.')
    args = parser.parse_args()

    generate_plots(args.directory, args.dataframe, args.plot)
