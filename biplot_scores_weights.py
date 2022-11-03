import argparse
from pathlib import Path
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

sns.set(style='whitegrid', palette='deep')


def get_weights(coeffs, x, y, bead):
    x_head = coeffs[f'PC{x + 1}'].filter(like=bead)
    y_head = coeffs[f'PC{y + 1}'].filter(like=bead)
    if len(x_head.index) <= 10:
        x_head = x_head
        y_head = y_head
    else:
        x_head = x_head.astype(float)
        y_head = y_head.astype(float)
        x_head_pos = x_head.nlargest(3)
        x_head_neg = x_head.nsmallest(3)
        x_head = x_head_pos.combine_first(x_head_neg)
        # y_head_pos = y_head.nlargest(3)
        # y_head_neg = y_head.nsmallest(3)
        y_head = y_head[x_head.index]
        # print(x_head, '\n', y_head)

    lipid_head = pd.concat({f'PC{x + 1}': x_head, f'PC{y + 1}': y_head}, axis=1)
    return lipid_head


def plot_biplot(scores, coeffs, dir_path, filename):
    fig = plt.figure(constrained_layout=True, figsize=(15, 5), dpi=150)
    gs = mpl.gridspec.GridSpec(nrows=1, ncols=3, figure=fig, left=0.02, bottom=0.02, right=0.98, top=None, wspace=None,
                               hspace=None, width_ratios=None, height_ratios=None)

    pc_comb = itertools.combinations(scores.columns.tolist(), 2)
    for idx, (x, y) in enumerate(pc_comb):
        x_pc = f'PC{x + 1}'
        y_pc = f'PC{y + 1}'
        # if idx <= 3:
        #     x_idx = 0
        #     y_idx = idx
        # elif 4 <= idx <= 7:
        #     x_idx = 1
        #     y_idx = idx - 4
        # else:
        #     x_idx = 2
        #     y_idx = idx - 8
        if idx <= 4:
            x_idx = 0
            y_idx = idx
        else:
            x_idx = 1
            y_idx = idx - 5
        print(x_idx, y_idx)
        xs = scores[x]
        ys = scores[y]
        scalex = 1.0 / (xs.max() - xs.min())
        scaley = 1.0 / (ys.max() - ys.min())
        ax = fig.add_subplot(gs[x_idx, y_idx])
        ax.scatter(xs * scalex, ys * scaley, c='gray', alpha=0.5)  # , c=y
        ax.set_xlabel(x_pc)
        ax.set_ylabel(y_pc)
        cl_head = get_weights(coeffs, x, y, 'Nda')
        pg_head = get_weights(coeffs, x, y, 'P4')
        for cl_interaction, pg_interaction in zip(cl_head.index, pg_head.index):
            plt.arrow(0, 0, cl_head.loc[cl_interaction, x_pc] * 10, cl_head.loc[cl_interaction, y_pc] * 10, color='r',
                      alpha=0.9)
            plt.text(cl_head.loc[cl_interaction, x_pc] * 10.15, cl_head.loc[cl_interaction, y_pc] * 10.15,
                     cl_interaction, color='r', weight='bold', ha='center', va='center')
            plt.arrow(0, 0, pg_head.loc[pg_interaction, x_pc] * 10, pg_head.loc[pg_interaction, y_pc] * 10, color='b',
                      alpha=0.9)
            plt.text(pg_head.loc[pg_interaction, x_pc] * 10.15, pg_head.loc[pg_interaction, y_pc] * 10.15,
                     pg_interaction, color='b', weight='bold', ha='center', va='center')
        # ax.set_aspect('equal', 'box')

    plt.savefig(dir_path / f'{filename}.pdf')
    # plt.show()


def load_data(directory, scores, coefficients, filename, interaction):
    scores = pd.read_pickle(directory / scores)
    if interaction == 'two-body':
        coeffs = pd.read_pickle(directory / coefficients[0])
    else:
        coeffs = pd.read_pickle(directory / coefficients[1])
    coeffs = coeffs[~(coeffs == 0.0).all(axis=1)]

    plot_biplot(scores[scores.columns[0:3]], coeffs, directory, filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Biplot')
    parser.add_argument('-dir', '--directory', type=Path, required=True, help='Path for saving plots')
    parser.add_argument('-s', '--scores', type=Path, required=True, help='Path to dataframe with principal components')
    parser.add_argument('-c', '--coefficients', type=Path, nargs='+', required=True,
                        help='Path to dataframe with pca weights.')
    parser.add_argument('-fn', '--filename', type=str, required=False, default='biplot_test',
                        help='name for generated plot')
    parser.add_argument('-ia', '--interaction', type=str, required=True, choices=['two-body', 'three-body'],
                        help='Select interaction type to plot')

    args = parser.parse_args()
    load_data(args.directory, args.scores, args.coefficients, args.filename, args.interaction)
