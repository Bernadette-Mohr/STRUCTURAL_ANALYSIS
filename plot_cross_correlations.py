import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors


def plot_cross_correlation(df, label, dir_path):
    min_ = df[label].min()
    max_ = df[label].max()
    if min_ < 0:
        center = 0.0
    else:
        center = ((max_ - min_) / 2) + min_
    # print(min_, center, max_)
    divnorm = colors.TwoSlopeNorm(vmin=min_, vcenter=center, vmax=max_)

    fig, axes = plt.subplots(2, 3, figsize=(8, 5), sharey='row', dpi=150)
    axes = axes.flatten()
    for idx, col in enumerate(df.iloc[:, 2:7]):
        sns.scatterplot(x=col, y=label, c=df[label], data=df, alpha=0.9, ax=axes[idx], cmap='coolwarm', norm=divnorm)
        reg = LinearRegression()
        reg.fit(np.vstack(df[col].values), np.vstack(df[label]))
        score = reg.score(np.vstack(df[col].values), np.vstack(df[label]))
        left, right = axes[idx].get_xlim()
        x_new = np.linspace(left, right, 100)
        y_new = reg.predict(x_new[:, np.newaxis])
        axes[idx].plot(x_new, y_new, c='k',
                       label=f"{r'$R^2$'}: {np.round(score, 3)}")
        axes[idx].legend()
        axes[idx].set_xlabel(f'PC {idx + 1}')
        axes[idx].set_ylabel(f'{label}')
    fig.delaxes(axes[-1])
    plt.tight_layout()
    plt.savefig(dir_path / f'pc_cross-correlation_{label}.pdf')
    # plt.show()


def load_data(dir_path, df_path, label):
    df = pd.read_pickle(df_path)
    print(df.columns)
    plot_cross_correlation(df, label, dir_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Create cross-correlation plots with specified label to identify principal '
                                     'components.')
    parser.add_argument('-dir', '--directory', type=Path, required=True, help='Directory the plot gets saved in.')
    parser.add_argument('-df', '--dataframe', type=Path, required=True, help='Dataframe with principal components and '
                                                                             'labels.')
    parser.add_argument('-l', '--label', type=str, required=True, help='Header of the dataframe column with the '
                                                                       'label that will be cross-correlated with the '
                                                                       'PCs.')
    args = parser.parse_args()
    load_data(args.directory, args.dataframe, args.label)
