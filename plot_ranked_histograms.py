import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from slatm_ranking_table import process_data


def get_difference_slatms(directory, deltaG, mbtypes):
    diff_df = process_data(directory, deltaG, mbtypes)
    # diff_df = diff_df.loc[:, (diff_df != 0).any(axis=0)]
    interactions = diff_df.drop(['round', 'solute'], axis=1)
    groups = interactions.groupby(by=interactions.columns, axis=1)
    # print(interactions)
    for interaction, bins in groups:
        print(interaction)
        if not :
            print(bins)
        else:
            print('Not found')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Blah')
    parser.add_argument('-dir', '--directory', type=Path, required=True)
    parser.add_argument('-dg', '--deltaG', type=Path, required=True)
    parser.add_argument('-mbt', '--mbtypes', type=Path, required=True)

    args = parser.parse_args()

    get_difference_slatms(args.directory, args.deltaG, args.mbtypes)
