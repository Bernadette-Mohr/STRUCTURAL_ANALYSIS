import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import itertools
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib import colors

sns.set(style='whitegrid', palette='deep')
cmap = sns.color_palette('bwr')


def kmeans_cluster(pca_df, n_components, n_clusters):
    X = pca_df.loc[:, [i for i in range(n_components)]]  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    KM = KMeans(n_clusters, n_init=50, max_iter=500, random_state=42)
    XKM = KM.fit_transform(X)
    print(KM.n_features_in_)
    df_km = pd.DataFrame(X)
    df_km['cluster'] = (KM.labels_ + 1).astype(str)
    df_km['selec'] = pca_df['selec'].tolist()
    df_km['sol'] = pca_df['sol'].tolist()

    return df_km


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


def get_largest(coeffs, x, y, n, interaction):
    x_head = coeffs[f'PC{x + 1}']
    y_head = coeffs[f'PC{y + 1}']
    x_head = x_head.astype(float)
    y_head = y_head.astype(float)
    if len(x_head.index) <= n:
        # print('foo')
        x_head = x_head
        y_head = y_head
    elif interaction == 'three-body':
        # print('bar')
        x_head_pos = x_head.nlargest(n)
        x_head_neg = x_head.nsmallest(n)
        x_head_xtmp = x_head_pos.combine_first(x_head_neg)
        y_head_tmp = y_head[x_head_xtmp.index]
        y_head_pos = y_head.nlargest(n)
        y_head_neg = y_head.nsmallest(n)
        y_head = y_head_pos.combine_first(y_head_neg)
        x_head_ytmp = x_head[y_head.index]
        x_head = x_head_xtmp.combine_first(x_head_ytmp)
        y_head = y_head.combine_first(y_head_tmp)
    else:
        # print('baz')
        x_head_pos = x_head.nlargest(n)
        x_head_neg = x_head.nsmallest(n)
        x_head = x_head_pos.combine_first(x_head_neg)
        # y_head_pos = y_head.nlargest(3)
        # y_head_neg = y_head.nsmallest(3)
        y_head = y_head[x_head.index]
        # print(x_head, '\n', y_head)

    lipid_head = pd.concat({f'PC{x + 1}': x_head, f'PC{y + 1}': y_head}, axis=1)
    return lipid_head


def plot_biplot(scores, coeffs, scaler, labels, interaction, descriptor, dir_path, kmeans=False):
    if descriptor == 'cluster':
        pass
    else:
        min_ = labels[descriptor].min()
        max_ = labels[descriptor].max()
        if min_ < 0:
            center = 0.0
        else:
            center = ((max_ - min_) / 2) + min_
        divnorm = colors.TwoSlopeNorm(vmin=min_, vcenter=center, vmax=max_)
        sm = plt.cm.ScalarMappable(cmap='seismic', norm=colors.TwoSlopeNorm(vmin=np.round(min_, 1), vcenter=center,
                                                                            vmax=np.round(max_, 1)))

    fig = plt.figure(constrained_layout=True, figsize=(15, 10), dpi=150)
    gs = mpl.gridspec.GridSpec(nrows=3, ncols=4, figure=fig, left=0.06, bottom=0.02, right=0.68, top=None, wspace=None,
                               hspace=None, width_ratios=None, height_ratios=None)
    pc_comb = itertools.combinations(scores.columns[0:5].tolist(), 2)
    for idx, (x, y) in enumerate(pc_comb):
        x_pc = f'PC{x + 1}'
        y_pc = f'PC{y + 1}'
        if idx <= 3:
            x_idx = 0
            y_idx = idx
        elif 4 <= idx <= 7:
            x_idx = 1
            y_idx = idx - 4
        else:
            x_idx = 2
            y_idx = idx - 8
        # if idx <= 4:
        #     x_idx = 0
        #     y_idx = idx
        # else:
        #     x_idx = 1
        #     y_idx = idx - 5
        xs = scores[x]
        ys = scores[y]
        scalex = 1.0 / (xs.max() - xs.min())
        scaley = 1.0 / (ys.max() - ys.min())
        ax = fig.add_subplot(gs[x_idx, y_idx])
        if kmeans:
            sns.scatterplot(data=scores, x=scores[x] * scalex, y=scores[y] * scaley, hue='cluster', alpha=0.4, ax=ax)
            ax.get_legend().remove()
        else:
            sns.scatterplot(data=scores, x=scores[x] * scalex, y=scores[y] * scaley, c=labels[descriptor], alpha=0.4,
                            ax=ax, cmap='seismic', norm=divnorm)
        ax.set_xlabel(x_pc)
        ax.set_ylabel(y_pc)
        # cl_head = get_weights(coeffs, x, y, 'Nda')
        # pg_head = get_weights(coeffs, x, y, 'P4')
        if interaction == 'three-body':
            biggest = get_largest(coeffs, x, y, 1, interaction)
        else:
            biggest = get_largest(coeffs, x, y, 3, interaction)
        for n_body in biggest.index:
        # for cl_n_body, pg_n_body in zip(cl_head.index, pg_head.index):
            plt.arrow(0, 0, biggest.loc[n_body, x_pc] * scaler[0], biggest.loc[n_body, y_pc] * scaler[0],
                      color='k', alpha=0.9)
            plt.text(biggest.loc[n_body, x_pc] * scaler[1], biggest.loc[n_body, y_pc] * scaler[1],
                     n_body, color='k', weight='bold', ha='center', va='center')
            # plt.arrow(0, 0, pg_head.loc[pg_n_body, x_pc] * 1, pg_head.loc[pg_n_body, y_pc] * 1, color='b',
            #           alpha=0.9)
            # plt.text(pg_head.loc[pg_n_body, x_pc] * 1.15, pg_head.loc[pg_n_body, y_pc] * 1.15,
            #          pg_n_body, color='b', weight='bold', ha='center', va='center')
        ax.set_aspect('equal', 'box')
    if kmeans:
        handles, labels = ax.get_legend_handles_labels()
        ax1 = fig.add_subplot(gs[2, 2])
        plt.axis('off')
        ax1.legend(handles, labels, loc="upper left", bbox_to_anchor=(-0.15, 1))
    else:
        ax = fig.add_subplot(gs[2, 2])
        plt.axis('off')
        plt.colorbar(sm, location='right', orientation='vertical', pad=-0.99, ax=ax)
    plt.savefig(dir_path / f'biplot_{interaction}_{descriptor}.pdf')
    # plt.show()


def load_data(directory, scores, coefficients, labels, interaction, descriptor, kmeans=False):
    scores = pd.read_pickle(directory / scores)
    labels = pd.read_pickle(directory / labels)
    coefficients = pd.read_pickle(directory / coefficients)
    n_one_body = len([idx for idx in coefficients.index if '-' not in idx])
    n_two_body = len([idx for idx in coefficients.index if len(idx.split('-')) == 2])
    # print(n_one_body, n_two_body)
    if interaction == 'two-body':
        coeffs = coefficients.iloc[n_one_body:n_two_body]
        scaler = [1.0, 1.15]
        # coeffs = coefficients.iloc[0:n_two_body]
    else:
        coeffs = coefficients.iloc[n_one_body + n_two_body:]
        scaler = [1.0, 1.15]
    coeffs = coeffs[~(coeffs == 0.0).all(axis=1)]
    # Add KMeans clustering labels to scores df
    if kmeans:
        scores = kmeans_cluster(scores, 5, 6)
        scores.to_pickle(f'{directory}/wghtd_avg_PCA_kmeans.pickle')
    # plot_biplot(scores[scores.columns[0:3]], coeffs, directory, filename)
    # plot_biplot(scores, coeffs, scaler, labels, interaction, descriptor, directory, kmeans)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Biplot')
    parser.add_argument('-dir', '--directory', type=Path, required=True, help='Path for saving plots')
    parser.add_argument('-s', '--scores', type=Path, required=True, help='Path to dataframe with principal components')
    parser.add_argument('-c', '--coefficients', type=Path, required=True, help='Path to dataframe with pca weights.')
    parser.add_argument('-l', '--labels', type=Path, required=True, help='Path to dataframe with solute labels.')
    # parser.add_argument('-fn', '--filename', type=str, required=False, default='biplot_test',
    #                     help='name for generated plot')
    parser.add_argument('-ia', '--interaction', type=str, required=True, choices=['two-body', 'three-body'],
                        help='Select interaction type to plot')
    parser.add_argument('-desc', '--descriptor', type=str, required=False, default='cluster',
                        help='Descriptor for coloring the PCA plots.')
    parser.add_argument('--kmeans', action='store_true', required=False,
                        help='True or False: Should Kmeans clustering be performed for coloring the PCA plots?')

    args = parser.parse_args()
    load_data(args.directory, args.scores, args.coefficients, args.labels, args.interaction, args.descriptor,
              args.kmeans)
