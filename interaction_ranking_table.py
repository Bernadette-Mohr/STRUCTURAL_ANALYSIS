import argparse
from pathlib import Path
import pandas as pd


def sort_interactions(one_df):
    cols = pd.MultiIndex.from_product([[pc for pc in one_df.columns[:-1]], ['interactions', 'weights']])
    sorted_df = pd.DataFrame(index=range(len(one_df.index)), columns=cols)

    for pc in one_df.columns[:-1]:
        sorted_ = one_df[['interactions', pc]].copy()
        sorted_[pc] = pd.to_numeric(sorted_[pc], errors='coerce')
        sorted_.rename(columns={pc: 'weights'}, inplace=True)
        sorted_.sort_values(by='weights', ascending=False, inplace=True)
        sorted_.reset_index(inplace=True, drop=True)
        sorted_.columns = pd.MultiIndex.from_product([[pc], sorted_.columns])
        sorted_df = pd.concat([sorted_df, sorted_], axis=1)

    sorted_df.dropna(how='any', inplace=True, axis=1)
    return sorted_df


def process_data(dir_path, one_body, two_body, three_body):  # , principal_component
    df = pd.DataFrame()
    one_df = pd.read_pickle(dir_path / one_body)
    two_df = pd.read_pickle(dir_path / two_body)
    two_df.index.names = ['interactions']
    two_df = two_df[~(two_df == 0.0).all(axis=1)]
    two_df.reset_index(inplace=True)
    two_df = two_df.reindex(two_df.columns[two_df.columns != 'interactions'].union(['interactions']), axis=1)
    three_df = pd.read_pickle(dir_path / three_body)
    three_df.index.names = ['interactions']
    three_df = three_df[~(three_df == 0.0).all(axis=1)]
    three_df.reset_index(inplace=True)
    three_df = three_df.reindex(three_df.columns[three_df.columns != 'interactions'].union(['interactions']), axis=1)
    one_df = sort_interactions(one_df)
    one_df['body'] = 'one_body'
    two_df = sort_interactions(two_df)
    two_df['body'] = 'two_body'
    three_df = sort_interactions(three_df)
    three_df['body'] = 'three_body'
    df = pd.concat([df, one_df, two_df, three_df], axis=0)
    df.set_index(['body', df.index], inplace=True)
    # save_path = dir_path / 'sorted_interactions.csv'
    # df.to_csv(save_path)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Sort interactions, save as table')
    parser.add_argument('-dir', '--directory', type=Path, required=True, help='Directory containing interaction '
                                                                              'dataframes. Plot will be saved there.')
    parser.add_argument('-one', '--one_body', type=Path, required=True, help='Df with one-body interactions.')
    parser.add_argument('-two', '--two_body', type=Path, required=True, help='Df with two-body interactions.')
    parser.add_argument('-three', '--three_body', type=Path, required=True, help='Df with three-body interactions.')
    # parser.add_argument('-pc', '--principal_component', type=str, required=True, help='Column name for weights to '
    #                                                                                   'analyze.')

    args = parser.parse_args()

    process_data(args.directory, args.one_body, args.two_body, args.three_body)  # , args.principal_component
