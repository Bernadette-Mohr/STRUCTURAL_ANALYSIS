import argparse
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np


def get_difference(cl_bead, pg_bead):
    return np.subtract(cl_bead, pg_bead)


def log_addition(data, alpha=0.0000001):
    return np.log(np.where(data < alpha, alpha, data))


def get_if_zeroes(beads_array):
    if np.all(beads_array == 0, axis=0):
        return False
    else:
        return True


def get_non_empty(beads_list):
    existing_beads = [bead for bead in beads_list if get_if_zeroes(bead)]
    return existing_beads


def calculate_avg(bead1, bead2, bead3, bead4, bead5):
    slatms = get_non_empty([bead1, bead2, bead3, bead4, bead5])
    beads_mean = np.mean(np.array(slatms), axis=0)
    return beads_mean


def get_reverse_mapping(mbtypes):
    """
    Generates mapping from each index of SLATM vector to relevant 'mbtype interaction'.
    The numbers in the function correspond to the number of bins for 2-body
    and 3-body interactions in the SLATM vectors. Those are determined by the
    grid spacing variable set in the function generate_slatm. For a CG system,
    the number of bins for 2-body interactions is 40, the number of bins for
    3-body interactions is 20.
    """

    a = 0
    for i in mbtypes:
        if len(i) == 1:
            a += 1
    b = a  # ==> a/b: number of bead types in the data set (14): 1-body interactions

    c = 0
    for i in mbtypes:
        if len(i) == 2:
            c += 1
    # print('c:', c)
    d = c * 40 + b
    # c: number of 2-body interactions
    # d: number of 2-body interactions x SLATM 2-body bin width + index offset
    # ==> index of SLATM vector where 3-body interactions end

    e = 0
    for i in mbtypes:
        if len(i) == 3:
            e += 1
    # print('e:', e)
    f = e * 20 + d
    # e: number of 3-body interactions
    # f: number of 3-body interactions x SLATM 3-body bin width + index offset
    # ==> index of SLATM vector where 3-body interactions end -> last index of vector

    # print('b, d, f:', b, d, f)

    new_reverse_map = defaultdict()

    for i in range(a):
        new_reverse_map[i] = mbtypes[i]

    i = a
    j = a
    # print('end 1-body i:', i)
    # print('end 1-body j:', j)

    while i < d:
        # print('d loop:', range(i, i + 41))
        for q in (range(i, i + 41)):
            # print(mbtypes[j])
            new_reverse_map[q] = mbtypes[j]
        i = i + 40
        j = j + 1
    # print(f'end 2-body i: {i} = 40 x c + b')
    # print('end 2-body j:', j)  # j - j_old = 119 - 14 = 105 = c

    while i < f:
        # print('f loop:', range(i, i + 21))
        # Diego set the range to the same bin width as the 2-body interactions
        for q in (range(i, i + 21)):
            # print(mbtypes[j])
            new_reverse_map[q] = mbtypes[j]
        new_reverse_map[i] = mbtypes[j]
        i = i + 20
        j = j + 1
    # print(f'end 3-body i: {i} = 20 x e + d')  # ==> i: len(slatm vector)
    # print(f'end 3-body j: {j}')  # ==> j: len(mbtypes)
    return new_reverse_map


def process_data(dir_path, delta_g, mbtypes):
    slatms_path = dir_path / 'SLATMS'
    df = pd.DataFrame()
    for cl, pg in zip(sorted(slatms_path.glob('SLATMS-CDL2-batch_*.pickle')),
                      sorted(slatms_path.glob('SLATMS-POPG-batch_*.pickle'))):
        cl_df = pd.read_pickle(cl)
        pg_df = pd.read_pickle(pg)
        merged_df = pd.merge(cl_df[['round', 'solute', 'bead1', 'bead2', 'bead3', 'bead4', 'bead5']],
                             pg_df[['round', 'solute', 'bead1', 'bead2', 'bead3', 'bead4', 'bead5']],
                             on=['round', 'solute'], how='inner', suffixes=('_cl', '_pg'))
        df = pd.concat([df, merged_df], ignore_index=True)
    df.sort_values(['round', 'solute'], ascending=[True, True], inplace=True,
                   ignore_index=True)
    deltaG = pd.read_pickle(dir_path / delta_g)
    mbt_df = pd.read_pickle(dir_path / mbtypes)
    mbtypes = mbt_df['mbtypes']
    charges = mbt_df['charges']
    mapping = mbt_df['mapping']

    # Maps location in vector to which mbtype interaction it represents
    new_reverse_map = get_reverse_mapping(mbtypes)
    # Maps mbtype number to specfic bead
    reverse_charges = {v: k for k, v in charges.items()}
    interactions = list()
    for idx, interaction in new_reverse_map.items():
        interactions.append("-".join([reverse_charges[i] for i in interaction]))

    df = pd.merge(df, deltaG[['round', 'solute', 'DeltaDeltaG']],
                  on=['round', 'solute'], how='inner')

    cl_df = df.filter(regex=r'_cl').copy()
    pg_df = df.filter(regex=r'_pg').copy()

    cl_df['cl_avg'] = cl_df.apply(lambda row: calculate_avg(row['bead1_cl'], row['bead2_cl'], row['bead3_cl'],
                                                            row['bead4_cl'], row['bead5_cl']), axis=1)
    pg_df['pg_avg'] = pg_df.apply(lambda row: calculate_avg(row['bead1_pg'], row['bead2_pg'], row['bead3_pg'],
                                                            row['bead4_pg'], row['bead5_pg']), axis=1)
    alpha = np.divide(1, np.power(10, np.divide(99, 10)))
    cl = log_addition(np.vstack(cl_df.cl_avg.values), alpha)
    pg = log_addition(np.vstack(pg_df.pg_avg.values), alpha)
    differences = get_difference(cl, pg)
    diff_df = pd.DataFrame(differences, columns=interactions[:-1])
    diff_df.insert(0, 'round', df['round'])
    diff_df.insert(1, 'solute', df['solute'])

    return diff_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Blah')
    parser.add_argument('-dir', '--directory', type=Path, required=True)
    parser.add_argument('-dg', '--deltaG', type=Path, required=True)
    parser.add_argument('-mbt', '--mbtypes', type=Path, required=True)

    args = parser.parse_args()

    process_data(args.directory, args.deltaG, args.mbtypes)
