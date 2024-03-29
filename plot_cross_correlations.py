# general functionalities
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

# plotting, analysis
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors

sns.set_context(context='paper', font_scale=1.5)


def style_label(label):
    # Styles y-axis labels for cross-correlation plots depending on which descriptor is used.
    if label == 'selectivity':
        deltaG = r"$\Delta\Delta G$"
        label = f'{label.capitalize()} {deltaG} [kcal/mol]'
    elif label == 'distance':
        norm = r"$L_2\mathrm{-norm}$"
        label = f'{norm}'
    elif label == 'hydrophobicity':
        dG_WOl = r"$\Delta G_{\mathrm{W}\rightarrow\mathrm{Ol}}$"
        label = f'average {dG_WOl} [kcal/mol]'
        # label = f'{label.capitalize()} {dG_w_ol}'
    elif label == 'h_bonds':
        label = f'avg. # H-bonds'
    elif label == 'polarity':
        label = f'avg. # polar beads'
    else:
        label = f'avg. # {label}'

    return label


def plot_cross_correlation(df, label, dir_path, pc=None):
    # Plot cross-correlations between principal components and a descriptor, either for all components in a data frame
    # or for a selected one.

    # Color scheme is selected depending on if the descriptor contains negative numbers or only positive numbers.
    min_ = df[label].min()
    max_ = df[label].max()
    if min_ < 0:
        center = 0.0
        cmap = 'coolwarm'
        divnorm = colors.TwoSlopeNorm(vmin=min_, vcenter=center, vmax=max_)
    else:
        cmap = 'flare'
        divnorm = None
    y_label = style_label(label)

    # If no principal component is passed, all of them are plotted.
    if not pc:
        fig, axes = plt.subplots(2, 3, figsize=(8, 5), sharey='row', dpi=150)
        axes = axes.flatten()
        for idx, col in enumerate(df.iloc[:, 2:8]):
            if min_ < 0:
                sns.scatterplot(x=col, y=label, c=df[label], data=df, alpha=0.9, ax=axes[idx], cmap=cmap, norm=divnorm)
            else:
                sns.scatterplot(x=col, y=label, c=df[label], data=df, alpha=0.9, ax=axes[idx], cmap=cmap)
            # Add linear regression line to provide an estimate of the linear correlation between descriptor and
            # principal components.
            reg = LinearRegression()
            reg.fit(np.vstack(df[col].values), np.vstack(df[label]))
            score = reg.score(np.vstack(df[col].values), np.vstack(df[label]))
            left, right = axes[idx].get_xlim()
            x_new = np.linspace(left, right, 100)
            y_new = reg.predict(x_new[:, np.newaxis])
            axes[idx].plot(x_new, y_new, c='k', label=f"{r'$R^2$'}: {np.round(score, 2)}")
            axes[idx].legend()
            axes[idx].set_xlabel(f'PC {idx + 1}')
            axes[idx].set_ylabel(f'{y_label}')
        plt.tight_layout()
        plt.savefig(dir_path / f'cross-correlation_wghtd-avg_{label}_6PCs.pdf', bbox_inches='tight')
    # If a principal component is passed, plot only a single cross-correlation.
    else:
        fig = plt.figure(dpi=150)
        ax = fig.add_subplot(111)
        if min_ < 0:
            sns.scatterplot(x=pc, y=label, c=df[label], data=df, alpha=0.9, ax=ax, cmap=cmap, norm=divnorm)
        else:
            sns.scatterplot(x=pc, y=label, c=df[label], data=df, alpha=0.9, ax=ax, cmap=cmap)
        reg = LinearRegression()
        reg.fit(np.vstack(df[pc].values), np.vstack(df[label]))
        score = reg.score(np.vstack(df[pc].values), np.vstack(df[label]))
        left, right = ax.get_xlim()
        x_new = np.linspace(left, right, 100)
        y_new = reg.predict(x_new[:, np.newaxis])
        ax.plot(x_new, y_new, c='k', label=f"{r'$R^2$'}: {np.round(score, 3)}")
        ax.legend()
        ax.set_xlabel(f'{pc}')
        ax.set_ylabel(f'{y_label}')
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
        # plt.tight_layout()
        plt.savefig(dir_path / f'cross-correlation_wghtd-avg_{label}_{pc}.pdf', bbox_inches='tight')
        # plt.show()


def load_data(dir_path, df_path, label, pc):
    # load dataframe, print columns to show names of different descriptors, plot cross-correlation and linear
    # regression line.
    df = pd.read_pickle(df_path)
    print(df.columns)
    plot_cross_correlation(df, label, dir_path, pc)


if __name__ == '__main__':
    # Handle command line inputs: destination directory for output, PCA data, label of the descriptor,
    # (principal component).
    parser = argparse.ArgumentParser('Create cross-correlation plots with specified label to identify principal '
                                     'components.')
    parser.add_argument('-dir', '--directory', type=Path, required=True, help='Directory the plot gets saved in.')
    parser.add_argument('-df', '--dataframe', type=Path, required=True, help='Dataframe with principal components and '
                                                                             'labels.')
    parser.add_argument('-l', '--label', type=str, required=True, help='Header of the dataframe column with the '
                                                                       'label that will be cross-correlated with the '
                                                                       'PCs.')
    parser.add_argument('-pc', '--principal', type=str, required=False, default=None,
                        help='OPTIONAL: Principal component to plot only a single correlation, for e.g. \'PC1\', '
                             '\'PC2\', \'PC3\', \'...\'. If not given all principal components are plotted in a grid.')
    args = parser.parse_args()
    load_data(args.directory, args.dataframe, args.label, args.principal)
