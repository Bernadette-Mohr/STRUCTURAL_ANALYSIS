import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sample_data(dataframe, directory):

    # empty dataframe for subsampled dataset
    batches = pd.DataFrame(columns=['batches', 'rounds', 'molecules', 'bins'])

    df = pd.read_pickle(dataframe)
    df['ΔΔG PG->CL'] = df['ΔΔG PG->CL'].astype(float)
    df.dropna(inplace=True)

    # quantile split data set into 10 roughly equal-numbered bins.
    df['bins'] = pd.qcut(df['ΔΔG PG->CL'], 10, retbins=False, labels=False)

    # sort data set by bins
    grouped = df.groupby('bins', dropna=True)
    for group in grouped:
        # randomize data in each bin
        shuffled = group[1].sample(frac=1)
        # split each bin into 4 equal subsets
        split = np.array_split(shuffled, 4)
        # insert label for each batch, append relevant columns to output dataframe
        for idx, part in enumerate(split):
            batch = f'batch_{str(idx)}'
            part.insert(loc=0, column='batches', value=batch)
            batches = pd.concat([batches, part[['batches', 'rounds', 'molecules', 'bins']]], ignore_index=True)
    # sort and write data to disc.
    batches.sort_values(by=['batches', 'rounds', 'molecules'], inplace=True, ignore_index=True)
    print(batches)
    file_path = directory / f'rerun_batches.pickle'
    batches.to_pickle(file_path)
    # df.hist(column='ΔΔG PG->CL')
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Subsample screened solutes by DeltaDeltaG.')
    parser.add_argument('-df', '--dataframe', type=Path, required=True, help='Pandas dataframe with screening results.')
    parser.add_argument('-dir', '--directory', type=Path, required=True, help='Where to store list with batches.')

    args = parser.parse_args()

    sample_data(args.dataframe, args.directory)
