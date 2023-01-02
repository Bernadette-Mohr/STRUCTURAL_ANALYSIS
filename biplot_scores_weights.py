import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import itertools

from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib import colors

from plot_cross_correlations import style_label

sns.set(style='whitegrid', palette='deep')
sns.set_context(context='paper', font_scale=1.8)


# cmap = sns.color_palette('bwr')


def kmeans_cluster(pca_df, n_components, n_clusters):
    X = pca_df.loc[:, [i for i in range(n_components)]]  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    KM = KMeans(n_clusters, n_init=50, max_iter=500, random_state=42)
    XKM = KM.fit_transform(X)
    # print(KM.n_features_in_)
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
        print('foo')
        x_head = x_head
        y_head = y_head
    elif interaction == 'three-body':
        print('bar')
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
        print('baz')
        x_head_pos = x_head.nlargest(n)
        x_head_neg = x_head.nsmallest(n)
        x_head = x_head_pos.combine_first(x_head_neg)
        # y_head_pos = y_head.nlargest(3)
        # y_head_neg = y_head.nsmallest(3)
        y_head = y_head[x_head.index]
        # print(x_head, '\n', y_head)

    lipid_head = pd.concat({f'PC{x + 1}': x_head, f'PC{y + 1}': y_head}, axis=1)
    return lipid_head


def plot_biplot(scores, coeffs, scaler, labels, interaction, descriptor, dir_path, pc=None, ts=None, kmeans=False):
    if descriptor == 'cluster':
        pass
    else:
        min_ = labels[descriptor].min()
        max_ = labels[descriptor].max()
        if min_ < 0:
            center = 0.0
            cmap = 'seismic'
            divnorm = colors.TwoSlopeNorm(vmin=min_, vcenter=center, vmax=max_)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=colors.TwoSlopeNorm(vmin=np.round(min_, 1), vcenter=center,
                                                                           vmax=np.round(max_, 1)))
        else:
            # center = ((max_ - min_) / 2) + min_
            cmap = 'flare'
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=colors.Normalize(vmin=np.round(min_, 1), vmax=np.round(max_, 1)))

    if not pc:
        fig = plt.figure(constrained_layout=True, figsize=(15, 14), dpi=150)
        gs = mpl.gridspec.GridSpec(nrows=4, ncols=4, figure=fig, left=0.06, bottom=0.02, right=0.68, top=0.58,
                                   wspace=0.01,
                                   hspace=0.01, width_ratios=[1, 1, 1, 1], height_ratios=[1, 1, 1, 1])
        pc_comb = itertools.combinations(scores.columns[0:6].tolist(), 2)
        for idx, (x, y) in enumerate(pc_comb):
            x_pc = f'PC{x + 1}'
            y_pc = f'PC{y + 1}'
            if idx <= 3:
                x_idx = 0
                y_idx = idx
            elif 4 <= idx <= 7:
                x_idx = 1
                y_idx = idx - 4
            elif 8 <= idx <= 11:
                x_idx = 2
                y_idx = idx - 8
            elif 12 <= idx <= 15:
                x_idx = 3
                y_idx = idx - 12
            else:
                x_idx = 4
                y_idx = idx - 16
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
                sns.scatterplot(data=scores, x=scores[x] * scalex, y=scores[y] * scaley, hue='cluster', alpha=0.4,
                                ax=ax)
                ax.get_legend().remove()
            elif min_ < 0.0:
                sns.scatterplot(data=scores, x=scores[x] * scalex, y=scores[y] * scaley, c=labels[descriptor],
                                alpha=0.4,
                                ax=ax, cmap=cmap, norm=divnorm)
            else:
                sns.scatterplot(data=scores, x=scores[x] * scalex, y=scores[y] * scaley, c=labels[descriptor],
                                alpha=0.4,
                                ax=ax, cmap=cmap)
            ax.set_xlabel(x_pc, fontsize=18)
            ax.set_ylabel(y_pc, fontsize=18)
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
                         n_body, color='k', weight='bold', ha='center', va='center', fontsize=14,
                         bbox=dict(boxstyle='square, pad=0.0', edgecolor=None, facecolor='w', alpha=0.5))
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
            desc_label = style_label(descriptor)
            ax = fig.add_subplot(gs[3, 3])
            plt.axis('off')
            cbar = plt.colorbar(sm, location='right', orientation='vertical', pad=-0.99, ax=ax)
            cbar.set_label(desc_label, rotation=270, labelpad=20, fontsize=18)
        plt.savefig(dir_path / f'biplot_{interaction}_{descriptor}_6PCs.pdf')
    else:
        fig = plt.figure(dpi=150)
        ax = fig.add_subplot(111)
        x_pc = pc[0]
        y_pc = pc[1]
        x = int(*filter(str.isdigit, x_pc)) - 1
        y = int(*filter(str.isdigit, y_pc)) - 1
        xs = scores[x]
        ys = scores[y]
        scalex = 1.0 / (xs.max() - xs.min())
        scaley = 1.0 / (ys.max() - ys.min())

        if min_ < 0.0:
            sns.scatterplot(data=scores, x=scores[x] * scalex, y=scores[y] * scaley, c=labels[descriptor], alpha=0.4,
                            ax=ax, cmap=cmap, norm=divnorm)
        else:
            sns.scatterplot(data=scores, x=scores[x] * scalex, y=scores[y] * scaley, c=labels[descriptor], alpha=0.4,
                            ax=ax, cmap=cmap)
        ax.set_xlabel(x_pc, fontsize=18)
        ax.set_ylabel(y_pc, fontsize=18)
        # cl_head = get_weights(coeffs, x, y, 'Nda')
        # pg_head = get_weights(coeffs, x, y, 'P4')
        if ts is not None:
            test_scores = pd.read_pickle(ts)
            test_xs = test_scores[x]
            test_ys = test_scores[y]
            test_scalex = 1.0 / (test_xs.max() - test_xs.min())
            test_scaley = 1.0 / (test_ys.max() - test_ys.min())
            sns.scatterplot(data=test_scores, x=test_scores[x] * test_scalex, y=test_scores[y] * test_scaley, color='k',
                            alpha=1.0, ax=ax)

            def plotlabel(xvar, yvar, label):
                plt.text(xvar, yvar + 0.05, label, color='k', ha='center', va='center', fontsize=11,
                         bbox=dict(boxstyle='square, pad=0.0', edgecolor=None, facecolor='w', alpha=0.5))

            test_scores.apply(lambda test: plotlabel(test[x] * test_scalex, test[y] * test_scaley, test['sol']), axis=1)

        else:
            if interaction == 'three-body':
                biggest = get_largest(coeffs, x, y, 1, interaction)
            else:
                biggest = get_largest(coeffs, x, y, 3, interaction)
            for n_body in biggest.index:
                # for cl_n_body, pg_n_body in zip(cl_head.index, pg_head.index):
                plt.arrow(0, 0, biggest.loc[n_body, x_pc] * scaler[0], biggest.loc[n_body, y_pc] * scaler[0],
                          color='k', alpha=0.9)
                plt.text(biggest.loc[n_body, x_pc] * scaler[1], biggest.loc[n_body, y_pc] * scaler[1],
                         n_body, color='k', weight='bold', ha='center', va='center', fontsize=14,
                         bbox=dict(boxstyle='square, pad=0.0', edgecolor=None, facecolor='w', alpha=0.5))

        ax.set_aspect('equal', 'box')

        desc_label = style_label(descriptor)
        cbar = plt.colorbar(sm, location='right', orientation='vertical', pad=.02, ax=ax)
        cbar.set_label(desc_label, rotation=270, labelpad=30, fontsize=18)

        plt.tight_layout()
        plt.savefig(dir_path / f'biplot_{interaction}_{descriptor}_{x_pc}vs{y_pc}_test-data.pdf')


def load_data(directory, scores, coefficients, labels, interaction, descriptor, pc=None, ts=None, kmeans=False):
    scores = pd.read_pickle(directory / scores)
    labels = pd.read_pickle(directory / labels)
    coefficients = pd.read_pickle(directory / coefficients)
    if ts is not None:
        ts_path = directory / ts
    else:
        ts_path = None
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
    plot_biplot(scores, coeffs, scaler, labels, interaction, descriptor, directory, pc, ts_path, kmeans)


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
    parser.add_argument('-pcs', '--principals', type=str, nargs=2, required=False, default=None,
                        help='Principal components, if only a single plot is to be generated.')
    parser.add_argument('-ts', '--test_scores', type=Path, required=False,
                        help='Path to dataframe with principal components of test data generated by pretrained model.')

    args = parser.parse_args()
    load_data(args.directory, args.scores, args.coefficients, args.labels, args.interaction, args.descriptor,
              args.principals, args.test_scores, args.kmeans)
