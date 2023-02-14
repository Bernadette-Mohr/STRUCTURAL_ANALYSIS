# general functionalities
import argparse
import itertools
from pathlib import Path
from string import ascii_lowercase

import numpy as np
import pandas as pd

# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import skunk
from matplotlib import colors
from matplotlib.offsetbox import AnnotationBbox
from plot_cross_correlations import style_label

sns.set(style='whitegrid', palette='deep')
sns.set_context(context='paper', font_scale=1.8)


def get_largest(coeffs, x, y, n):
    # Get the largest absolute eigenvector coefficients of the first principal component and the corresponding values
    # for the second principal component, and vice versa.
    # Return:
    #   lipid_coeff (pandas dataframe): two columns, largest absolute eigenvector coefficients in one of a pair of
    #                                   principal components and the corresponding value in the other.
    x_coeff = coeffs[f'PC{x + 1}']
    y_coeff = coeffs[f'PC{y + 1}']
    x_coeff = x_coeff.astype(float)
    y_coeff = y_coeff.astype(float)
    if len(x_coeff.index) <= n:
        x_coeff = x_coeff
        y_coeff = y_coeff
    else:
        x_coeff_pos = x_coeff.nlargest(n)
        x_coeff_neg = x_coeff.nsmallest(n)
        x_coeff_xtmp = x_coeff_pos.combine_first(x_coeff_neg)
        y_coeff_tmp = y_coeff[x_coeff_xtmp.index]
        y_coeff_pos = y_coeff.nlargest(n)
        y_coeff_neg = y_coeff.nsmallest(n)
        y_coeff = y_coeff_pos.combine_first(y_coeff_neg)
        x_coeff_ytmp = x_coeff[y_coeff.index]
        x_coeff = x_coeff_xtmp.combine_first(x_coeff_ytmp)
        y_coeff = y_coeff.combine_first(y_coeff_tmp)

    lipid_coeff = pd.concat({f'PC{x + 1}': x_coeff, f'PC{y + 1}': y_coeff}, axis=1)
    return lipid_coeff


def plotlabel(xvar, yvar, label):
    # Adds the interaction label in a box with white background to the end of the corresponding eigenvector
    # coefficients.
    plt.text(xvar, yvar + 0.05, label, color='k', ha='center', va='center', fontsize=11,
             bbox=dict(boxstyle='square, pad=0.0', edgecolor=None, facecolor='w', alpha=0.5))


def plotimage(xvar, yvar, label, letter_dict, img_size, ax):
    # If test solutes are transformed on a pretrained PCA model, images of their graph structures are added to the
    # resulting lower-dimensional points.
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
    # Pairwise visualization of principal components, colored by a descriptor of choice, and annotated with the main
    # absolute 2-body or 3-body interactions, or with graph images of test solutes.

    # Upper and lower bound for color gradient according to selected descriptor.
    min_ = labels[descriptor].min()
    max_ = labels[descriptor].max()
    # If the values of the selected descriptor can be negative, use diverging palette and center around zero.
    if min_ < 0:
        center = 0.0
        cmap = 'seismic'
        divnorm = colors.TwoSlopeNorm(vmin=min_, vcenter=center, vmax=max_)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=colors.TwoSlopeNorm(vmin=np.round(min_, 1), vcenter=center,
                                                                       vmax=np.round(max_, 1)))
    # If values are only positive, select sequential gradient.
    else:
        cmap = 'flare'
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=colors.Normalize(vmin=np.round(min_, 1), vmax=np.round(max_, 1)))

    filename = ''
    # If no principal components are passed from command line, all possible pairs are plotted in a grid.
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
            # Principal component scores are scaled to the interval [-1, 1] for plotting (same scale as eigenvector
            # coefficients).
            scalex = 1.0 / (xs.max() - xs.min())
            scaley = 1.0 / (ys.max() - ys.min())
            ax = fig.add_subplot(gs[x_idx, y_idx])

            # Handle coloring of the lower-dimensional samples according to the selected descriptor.
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

            # Get n largest absolute eigenvector coefficients from each of the principal components.
            if interaction == 'three-body':
                biggest = get_largest(coeffs, x, y, 1)
            else:
                biggest = get_largest(coeffs, x, y, 2)

            # Plot lines from the center of the axes to the point defined by the pair of eigenvector coefficients. Add
            # labels with the corresponding many-body interaction.
            for n_body in biggest.index:
                plt.arrow(0, 0, biggest.loc[n_body, x_pc] * scaler[0], biggest.loc[n_body, y_pc] * scaler[0],
                          color='k', alpha=0.9)
                plt.text(biggest.loc[n_body, x_pc] * scaler[1], biggest.loc[n_body, y_pc] * scaler[1],
                         n_body, color='k', weight='bold', ha='center', va='center', fontsize=14,
                         bbox=dict(boxstyle='square, pad=0.0', edgecolor=None, facecolor='w', alpha=0.5))
            ax.set_aspect('equal', 'box')

        # Insert a label for the descriptor gradient in the last grid.
        else:
            desc_label = style_label(descriptor)
            ax = fig.add_subplot(gs[3, 3])
            plt.axis('off')
            cbar = plt.colorbar(sm, location='right', orientation='vertical', pad=-0.99, ax=ax)
            cbar.set_label(desc_label, rotation=270, labelpad=20, fontsize=18)

        filename = f'biplot_{interaction}_{descriptor}_6PCs.pdf'
        # plt.savefig(dir_path / f'biplot_{interaction}_{descriptor}_6PCs.pdf')

    # If a specific pair of principal components is selected via the command line to generate a single biplot:
    else:
        fig = plt.figure(dpi=150)
        ax = fig.add_subplot(111)
        x_pc = pc[0]
        y_pc = pc[1]
        x = int(*filter(str.isdigit, x_pc)) - 1
        y = int(*filter(str.isdigit, y_pc)) - 1

        xs = scores[x]
        ys = scores[y]
        # Principal component scores are scaled to the interval [-1, 1] for plotting (same scale as eigenvector
        # coefficients).
        scalex = 1.0 / (xs.max() - xs.min())
        scaley = 1.0 / (ys.max() - ys.min())

        # Handle coloring of the lower-dimensional samples according to the selected descriptor.
        if min_ < 0.0:
            sns.scatterplot(data=scores, x=scores[x] * scalex, y=scores[y] * scaley, c=labels[descriptor], alpha=0.4,
                            ax=ax, cmap=cmap, norm=divnorm)
        else:
            sns.scatterplot(data=scores, x=scores[x] * scalex, y=scores[y] * scaley, c=labels[descriptor], alpha=0.4,
                            ax=ax, cmap=cmap)

        # Add diagonal dasshed line to highlight the linear relationship between the two plotted principal components
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.plot([xlim[0], xlim[1]], [ylim[1], ylim[0]], linestyle='dashed', color='gray')
        ax.set_xlabel(x_pc, fontsize=18)
        ax.set_ylabel(y_pc, fontsize=18)

        # If test solutes are passed, add their lower-dimensional points to the biplot.
        if ts is not None:
            test_scores = pd.read_pickle(ts)
            test_xs = test_scores[x]
            test_ys = test_scores[y]
            # Principal component scores are scaled to the interval [-1, 1] for plotting (same scale as eigenvector
            # coefficients).
            test_scalex = 1.0 / (test_xs.max() - test_xs.min())
            test_scaley = 1.0 / (test_ys.max() - test_ys.min())

            # Add lower-dimensional points of the test solutes to the biplot
            sns.scatterplot(data=test_scores, x=test_scores[x] * test_scalex, y=test_scores[y] * test_scaley, color='k',
                            alpha=1.0, ax=ax)

            # If no image files are passed for plotting, plot labels with identifier of each solute
            if images is None:
                test_scores.apply(lambda test: plotlabel(test[x] * test_scalex, test[y] * test_scaley, test['sol']),
                                  axis=1)
                filename = f'biplot_{descriptor}_{x_pc}vs{y_pc}_test-data.pdf'
            # If image files are passed through command line, add them to the plot as inserts.
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

                # Add color bar for the selected descriptor with the appropriate color scheme.
                desc_label = style_label(descriptor)
                cbar = plt.colorbar(sm, location='right', orientation='vertical', pad=.02, ax=ax)
                cbar.set_label(desc_label, rotation=270, labelpad=30, fontsize=18)

                plt.tight_layout()
                test_scores.apply(lambda test: plotimage(test[x] * test_scalex, test[y] * test_scaley, test['sol'],
                                                         letters_dict, img_size, ax=ax), axis=1)
                svg = skunk.insert(image_dict)
                filename = f'biplot_{descriptor}_{x_pc}vs{y_pc}_test-data_images.svg'

        # If a pair of principal components, but no transformed SLATM representations for test solutes are passed,
        # generate a single biplot with eigenvector coefficients and interaction labels.
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

        # Differences in styling and writing to file, depending on whether solute images are inserted or not.
        if images:
            svg_path = dir_path / filename
            with open(svg_path, 'w') as svgfile:
                svgfile.write(svg)
        else:
            ax.set_aspect('equal', 'box')

            desc_label = style_label(descriptor)
            cbar = plt.colorbar(sm, location='right', orientation='vertical', pad=.02, ax=ax)
            cbar.set_label(desc_label, rotation=270, labelpad=30, fontsize=18)

            plt.tight_layout()
            plt.savefig(dir_path / filename)


def load_data(directory, scores, coefficients, labels, interaction, descriptor, pc=None, ts=None, images=None):
    # Load principal components, descriptors and eigenvectors for the training set PCA as pandas dataframes, same data
    # for test solutes when when passed. Identify number of 1-body and 2-body interactions in order to select
    # corresponding values.
    scores = pd.read_pickle(directory / scores)
    labels = pd.read_pickle(directory / labels)
    coefficients = pd.read_pickle(directory / coefficients)

    # If test solute principal components are passed, set their path.
    if ts is not None:
        ts_path = directory / ts
    else:
        ts_path = None
    n_one_body = len([idx for idx in coefficients.index if '-' not in idx])
    n_two_body = len([idx for idx in coefficients.index if len(idx.split('-')) == 2])

    # Extract 2-body or 3-body interactions, scaler handles label placement for interaction labels.
    if interaction == 'two-body':
        coeffs = coefficients.iloc[n_one_body:n_two_body]
        scaler = [1.0, 1.15]
    else:
        coeffs = coefficients.iloc[n_one_body + n_two_body:]
        scaler = [1.0, 1.15]
    coeffs = coeffs[~(coeffs == 0.0).all(axis=1)]

    plot_biplot(scores, coeffs, scaler, labels, interaction, descriptor, directory, images, pc, ts_path)


if __name__ == '__main__':
    # Handles passing of PCA results, image files and setting of plotting flags via command line.
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
                        help='Paths to files with graph images in SVG format for inserting images instead of solute '
                             'labels.')

    # -ts weighted_average_PCA_6PCs_test-data.pickle
    # -ims /home/bernadette/Documents/STRUCTURAL_ANALYSIS/ROUND_6_molecule_48.svg
    # /home/bernadette/Documents/STRUCTURAL_ANALYSIS/ROUND_7_molecule_34.svg
    # /home/bernadette/Documents/STRUCTURAL_ANALYSIS/ROUND_7_molecule_56.svg

    args = parser.parse_args()
    load_data(args.directory, args.scores, args.coefficients, args.labels, args.interaction, args.descriptor,
              args.principals, args.test_scores, args.images)
