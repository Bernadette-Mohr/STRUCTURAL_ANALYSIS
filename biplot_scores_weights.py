import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import itertools

from matplotlib.offsetbox import AnnotationBbox
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
import seaborn as sns
import skunk
from string import ascii_lowercase

from plot_cross_correlations import style_label

sns.set(style='whitegrid', palette='deep')
sns.set_context(context='paper', font_scale=1.8)


def get_weights(coeffs, x, y, bead):
    x_head = coeffs[f'PC{x + 1}'].filter(like=bead)
    y_head = coeffs[f'PC{y + 1}'].filter(like=bead)
    if len(x_head.index) <= 10:
        x_head = x_head
        y_head = y_head
    else:
        x_head = x_head.astype(float)
        y_head = y_head.astype(float)
        x_head_pos = x_head.nlargest(20)
        x_head_neg = x_head.nsmallest(20)
        x_head = x_head_pos.combine_first(x_head_neg)
        y_head = y_head[x_head.index]

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
        y_head = y_head[x_head.index]

    lipid_head = pd.concat({f'PC{x + 1}': x_head, f'PC{y + 1}': y_head}, axis=1)
    return lipid_head


def plotlabel(xvar, yvar, label):
    plt.text(xvar, yvar + 0.05, label, color='k', ha='center', va='center', fontsize=11,
             bbox=dict(boxstyle='square, pad=0.0', edgecolor=None, facecolor='w', alpha=0.5))


def plotimage(xvar, yvar, label, letter_dict, img_size, ax):
    box = skunk.Box(int(img_size[0] * 0.1), int(img_size[1] * 0.1), label)
    transform = ax.transData.transform((xvar, yvar))
    xdisplay, ydisplay = ax.transAxes.inverted().transform(transform)
    xpad, ypad = 0.05, 0.1
    ab = AnnotationBbox(box, (xvar, yvar),
                        xybox=(xdisplay + xpad, ydisplay + ypad),
                        xycoords='data',
                        boxcoords=('axes fraction', 'axes fraction'),
                        pad=0.0,
                        bboxprops=dict(alpha=0.0),
                        arrowprops=dict(arrowstyle='-|>, head_width=0.1', color='gray', lw=1.0)
                        )
    ax.add_artist(ab)
    ax.text(xvar + 0.32, yvar + 0.26, letter_dict[label], color='k', ha='center', va='center', fontsize=14)


def plot_biplot(scores, coeffs, scaler, labels, interaction, descriptor, dir_path, images, pc=None, ts=None):
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
    filename = ''
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

            xs = scores[x]
            ys = scores[y]
            scalex = 1.0 / (xs.max() - xs.min())
            scaley = 1.0 / (ys.max() - ys.min())
            ax = fig.add_subplot(gs[x_idx, y_idx])
            if min_ < 0.0:
                sns.scatterplot(data=scores, x=scores[x] * scalex, y=scores[y] * scaley, c=labels[descriptor],
                                alpha=0.4,
                                ax=ax, cmap=cmap, norm=divnorm)
            else:
                sns.scatterplot(data=scores, x=scores[x] * scalex, y=scores[y] * scaley, c=labels[descriptor],
                                alpha=0.4,
                                ax=ax, cmap=cmap)
            ax.set_xlabel(x_pc, fontsize=18)
            ax.set_ylabel(y_pc, fontsize=18)
            if interaction == 'three-body':
                biggest = get_largest(coeffs, x, y, 1, interaction)
            else:
                biggest = get_largest(coeffs, x, y, 3, interaction)
            for n_body in biggest.index:
                plt.arrow(0, 0, biggest.loc[n_body, x_pc] * scaler[0], biggest.loc[n_body, y_pc] * scaler[0],
                          color='k', alpha=0.9)
                plt.text(biggest.loc[n_body, x_pc] * scaler[1], biggest.loc[n_body, y_pc] * scaler[1],
                         n_body, color='k', weight='bold', ha='center', va='center', fontsize=14,
                         bbox=dict(boxstyle='square, pad=0.0', edgecolor=None, facecolor='w', alpha=0.5))
            ax.set_aspect('equal', 'box')
        else:
            desc_label = style_label(descriptor)
            ax = fig.add_subplot(gs[3, 3])
            plt.axis('off')
            cbar = plt.colorbar(sm, location='right', orientation='vertical', pad=-0.99, ax=ax)
            cbar.set_label(desc_label, rotation=270, labelpad=20, fontsize=18)
        filename = f'biplot_{interaction}_{descriptor}_6PCs.pdf'
        # plt.savefig(dir_path / f'biplot_{interaction}_{descriptor}_6PCs.pdf')
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
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.plot([xlim[0], xlim[1]], [ylim[1], ylim[0]], linestyle='dashed', color='gray')
        ax.set_xlabel(x_pc, fontsize=18)
        ax.set_ylabel(y_pc, fontsize=18)

        if ts is not None:
            test_scores = pd.read_pickle(ts)
            test_xs = test_scores[x]
            test_ys = test_scores[y]
            test_scalex = 1.0 / (test_xs.max() - test_xs.min())
            test_scaley = 1.0 / (test_ys.max() - test_ys.min())
            sns.scatterplot(data=test_scores, x=test_scores[x] * test_scalex, y=test_scores[y] * test_scaley, color='k',
                            alpha=1.0, ax=ax)

            if images is None:
                test_scores.apply(lambda test: plotlabel(test[x] * test_scalex, test[y] * test_scaley, test['sol']),
                                  axis=1)
                filename = f'biplot_{descriptor}_{x_pc}vs{y_pc}_test-data.pdf'
            else:
                img_size = fig.get_size_inches() * fig.dpi
                image_dict = dict()
                letters_dict = dict()
                for idx, label in enumerate(test_scores['sol'].tolist()):
                    letters_dict[label] = f'({ascii_lowercase[idx]})'
                    for image in images:
                        name = image.stem
                        if label.replace(' ', '_') in name:
                            image_dict[label] = str(image)
                            break
                ax.set_aspect('equal', 'box')

                desc_label = style_label(descriptor)
                cbar = plt.colorbar(sm, location='right', orientation='vertical', pad=.02, ax=ax)
                cbar.set_label(desc_label, rotation=270, labelpad=30, fontsize=18)

                plt.tight_layout()
                test_scores.apply(lambda test: plotimage(test[x] * test_scalex, test[y] * test_scaley, test['sol'],
                                                         letters_dict, img_size, ax=ax), axis=1)
                svg = skunk.insert(image_dict)
                filename = f'biplot_{descriptor}_{x_pc}vs{y_pc}_test-data_images.svg'

        else:
            if interaction == 'three-body':
                biggest = get_largest(coeffs, x, y, 1, interaction)
            else:
                biggest = get_largest(coeffs, x, y, 3, interaction)
            for n_body in biggest.index:
                plt.arrow(0, 0, biggest.loc[n_body, x_pc] * scaler[0], biggest.loc[n_body, y_pc] * scaler[0],
                          color='k', alpha=0.9)
                plt.text(biggest.loc[n_body, x_pc] * scaler[1], biggest.loc[n_body, y_pc] * scaler[1],
                         n_body, color='k', weight='bold', ha='center', va='center', fontsize=14,
                         bbox=dict(boxstyle='square, pad=0.0', edgecolor=None, facecolor='w', alpha=0.5))
                filename = f'biplot_{interaction}_{descriptor}_{x_pc}vs{y_pc}.pdf'

        if images is None:
            ax.set_aspect('equal', 'box')

            desc_label = style_label(descriptor)
            cbar = plt.colorbar(sm, location='right', orientation='vertical', pad=.02, ax=ax)
            cbar.set_label(desc_label, rotation=270, labelpad=30, fontsize=18)

            plt.tight_layout()
            plt.savefig(dir_path / filename)
        else:
            svg_path = dir_path / filename
            with open(svg_path, 'w') as svgfile:
                svgfile.write(svg)


def load_data(directory, scores, coefficients, labels, interaction, descriptor, pc=None, ts=None, images=None):
    scores = pd.read_pickle(directory / scores)
    labels = pd.read_pickle(directory / labels)
    coefficients = pd.read_pickle(directory / coefficients)
    if ts is not None:
        ts_path = directory / ts
    else:
        ts_path = None
    n_one_body = len([idx for idx in coefficients.index if '-' not in idx])
    n_two_body = len([idx for idx in coefficients.index if len(idx.split('-')) == 2])
    if interaction == 'two-body':
        coeffs = coefficients.iloc[n_one_body:n_two_body]
        scaler = [1.0, 1.15]
    else:
        coeffs = coefficients.iloc[n_one_body + n_two_body:]
        scaler = [1.0, 1.15]
    coeffs = coeffs[~(coeffs == 0.0).all(axis=1)]

    plot_biplot(scores, coeffs, scaler, labels, interaction, descriptor, directory, images, pc, ts_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Biplot')
    parser.add_argument('-dir', '--directory', type=Path, required=True, help='Path for saving plots')
    parser.add_argument('-s', '--scores', type=Path, required=True, help='Path to dataframe with principal components')
    parser.add_argument('-c', '--coefficients', type=Path, required=True, help='Path to dataframe with pca weights.')
    parser.add_argument('-l', '--labels', type=Path, required=True, help='Path to dataframe with solute labels.')
    parser.add_argument('-ia', '--interaction', type=str, required=True, choices=['two-body', 'three-body'],
                        help='Select interaction type to plot')
    parser.add_argument('-desc', '--descriptor', type=str, required=False, default='cluster',
                        help='Descriptor for coloring the PCA plots.')
    parser.add_argument('-pcs', '--principals', type=str, nargs=2, required=False, default=None,
                        help='Principal components, if only a single plot is to be generated.')
    parser.add_argument('-ts', '--test_scores', type=Path, required=False,
                        help='Path to dataframe with principal components of test data generated by pretrained model.')
    parser.add_argument('-ims', '--images', type=Path, required=False, nargs='+', default=None,
                        help='Paths to svg files with graph images for inserting images instead of solute labels.')

    # -ts weighted_average_PCA_6PCs_test-data.pickle
    # -ims /home/bernadette/Documents/STRUCTURAL_ANALYSIS/ROUND_6_molecule_48.svg
    # /home/bernadette/Documents/STRUCTURAL_ANALYSIS/ROUND_7_molecule_34.svg
    # /home/bernadette/Documents/STRUCTURAL_ANALYSIS/ROUND_7_molecule_56.svg

    args = parser.parse_args()
    load_data(args.directory, args.scores, args.coefficients, args.labels, args.interaction, args.descriptor,
              args.principals, args.test_scores, args.images)
