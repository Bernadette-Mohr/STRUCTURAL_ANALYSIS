import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from interaction_ranking_table import process_data

sns.set(style='white', palette='deep')


def plot_interactions(pos, neg, ax):
    interaction = pos.index.get_level_values(0).tolist()[0]
    pos = pos.droplevel('body')
    pos = pos.iloc[::-1]
    neg = neg.droplevel('body')
    ax2 = ax.twiny()
    if interaction == 'one_body':
        pos.plot(x='interactions', y='weights', legend=False, color='g', style='.--', xticks=range(len(pos.index)),
                 ax=ax2)
        neg.plot(x='interactions', y='weights', legend=False, grid=True, color='r', style='.--',
                 xticks=range(len(neg.index)), ax=ax)
    else:
        pos.plot(x='interactions', y='weights', legend=False, rot=30, color='g', style='.--',
                 xticks=range(len(pos.index)), ax=ax2)
        neg.plot(x='interactions', y='weights', legend=False, grid=True, rot=30, color='r', style='.--',
                 xticks=range(len(neg.index)), ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('weights')
    ax2.set(xlabel=None)
    ax2.set_ylabel('weights')


def load_data(dir_path, one_body, two_body, three_body, principal_component):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(7, 9), dpi=150)
    df = process_data(dir_path, one_body, two_body, three_body)
    groups = df.groupby(level='body', sort=False)
    for idx, (interaction, group) in enumerate(groups):
        pc = group[principal_component]
        pos = pc[pc['weights'] > 0.0]
        neg = pc[pc['weights'] < 0.0]
        if len(pos.index) < 10:
            pos = pos
        else:
            pos = pos.nlargest(10, ['weights'])
        if len(neg.index) < 10:
            neg = neg.iloc[neg.weights.abs().argsort()]
        else:
            neg = neg.iloc[neg.weights.abs().argsort()].tail(10)

        plot_interactions(pos, neg, axes[idx])

    # plt.tight_layout()
    axes[0].set_title(principal_component)
    plt.subplots_adjust(hspace=0.9, left=0.14, right=0.98, top=0.93, bottom=0.09)
    path = dir_path / f'{principal_component}_ranked_weights.pdf'
    plt.savefig(path)
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Sort interactions, save as table')
    parser.add_argument('-dir', '--directory', type=Path, required=True, help='Directory containing interaction '
                                                                              'dataframes. Plot will be saved there.')
    parser.add_argument('-one', '--one_body', type=Path, required=True, help='Df with one-body interactions.')
    parser.add_argument('-two', '--two_body', type=Path, required=True, help='Df with two-body interactions.')
    parser.add_argument('-three', '--three_body', type=Path, required=True, help='Df with three-body interactions.')
    parser.add_argument('-pc', '--principal_component', type=str, required=True, help='Column name for weights to '
                                                                                      'analyze.')

    args = parser.parse_args()

    load_data(args.directory, args.one_body, args.two_body, args.three_body, args.principal_component)
