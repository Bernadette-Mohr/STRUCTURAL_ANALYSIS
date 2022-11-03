import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import seaborn as sns
from interaction_ranking_table import process_data

sns.set(style='white', palette='deep')
pal = sns.color_palette('deep').as_hex()
# print(pal)


# plt.rcParams['axes.axisbelow'] = True

def plot_interactions(pos_cl, pos_pg, neg_cl, neg_pg, ax_top, ax_bottom):
    interaction = pos_cl.index.get_level_values(0).tolist()[0]
    pos_cl = pos_cl.droplevel('body')
    pos_cl = pos_cl.iloc[::-1]
    pos_pg = pos_pg.droplevel('body')
    pos_pg = pos_pg.iloc[::-1]
    neg_cl = neg_cl.droplevel('body')
    neg_cl['weights'] = neg_cl['weights'].abs()
    neg_pg = neg_pg.droplevel('body')
    neg_pg['weights'] = neg_pg['weights'].abs()
    ax2 = ax_top.twiny()
    ax2b = ax_top.twiny()
    ax3 = ax_bottom.twiny()
    ax3b = ax_bottom.twiny()
    if interaction == 'one_body':
        pos_cl.plot(x='interactions', y='weights', legend=False, color='g', style='.--',
                    xticks=range(len(pos_cl.index)), ax=ax2)
        pos_pg.plot(x='interactions', y='weights', legend=False, color='g', style='.--',
                    xticks=range(len(pos_pg.index)), ax=ax_top)
        neg_cl.plot(x='interactions', y='weights', legend=False, grid=True, color='r', style='.--',
                    xticks=range(len(neg_cl.index)), ax=ax3)
        neg_pg.plot(x='interactions', y='weights', legend=False, grid=True, color='r', style='.--',
                    xticks=range(len(neg_pg.index)), ax=ax_bottom)
    else:
        pos_cl.plot(x='interactions', y='weights', legend=True, rot=30, color='r', style='.--',
                    xticks=range(len(pos_cl.index)), ax=ax2, grid=True)
        pos_pg.plot(x='interactions', y='weights', legend=False, rot=30, color='b', style='.--',
                    xticks=range(len(pos_pg.index)), ax=ax_top, grid=True)
        pos_pg.plot(x='interactions', y='weights', legend=False, rot=30, color='b', style='.--',
                    xticks=range(len(pos_pg.index)), ax=ax2b, grid=False)
        neg_cl.plot(x='interactions', y='weights', legend=True, rot=30, color='r', style='.--',
                    xticks=range(len(neg_cl.index)), ax=ax3, grid=True)
        neg_pg.plot(x='interactions', y='weights', legend=False, rot=30, color='b', style='.--',
                    xticks=range(len(neg_pg.index)), ax=ax_bottom, grid=True)
        neg_pg.plot(x='interactions', y='weights', legend=False, rot=30, color='b', style='.--',
                    xticks=range(len(neg_pg.index)), ax=ax3b, grid=False)

    blue_line = mlines.Line2D([], [], linestyle='-.', color=pal[0], label='PG: P4')
    red_line = mlines.Line2D([], [], linestyle='-.', color=pal[3], label='CL: Nda')
    ax_top.set_xlabel('')
    ax_top.set_ylabel('weights')
    ax2.set(xlabel=None)
    ax2b.set(xlabel=None)
    ax2b.get_xaxis().set_visible(False)
    ax2b.get_yaxis().set_visible(False)
    ax2.legend(handles=[red_line, blue_line], loc='lower right')
    ax2.set_ylabel('weights')

    ax_bottom.set_xlabel('')
    ax_bottom.set_ylabel('weights')
    ax_bottom.invert_yaxis()
    ax3.set(xlabel=None)
    ax3b.set(xlabel=None)
    ax3b.get_xaxis().set_visible(False)
    ax3b.get_yaxis().set_visible(False)
    ax3.legend(handles=[red_line, blue_line], loc='best')
    ax3.set_ylabel('weights')


def separate_members(part_df):
    cl = part_df[part_df['interactions'].str.contains('Nda')]
    pg = part_df[part_df['interactions'].str.contains('P4')]
    qa = part_df[(part_df['interactions'].str.contains('Qa')) & (~part_df['interactions'].str.contains('Nda'))
                 & (~part_df['interactions'].str.contains('P4'))]
    na = part_df[part_df['interactions'].str.contains('Na') & (~part_df['interactions'].str.contains('Nda'))
                 & (~part_df['interactions'].str.contains('P4'))]
    c1 = part_df[part_df['interactions'].str.contains('C1') & (~part_df['interactions'].str.contains('Nda'))
                 & (~part_df['interactions'].str.contains('P4'))]
    c3 = part_df[part_df['interactions'].str.contains('C3') & (~part_df['interactions'].str.contains('Nda'))
                 & (~part_df['interactions'].str.contains('P4'))]
    water = part_df[part_df['interactions'].str.contains('POL') & (~part_df['interactions'].str.contains('Nda'))
                    & (~part_df['interactions'].str.contains('P4'))]
    ion = part_df[part_df['interactions'].str.contains('PQd') & (~part_df['interactions'].str.contains('Nda'))
                  & (~part_df['interactions'].str.contains('P4'))]
    other = part_df[(~part_df['interactions'].str.contains('Nda')) & (~part_df['interactions'].str.contains('P4'))
                    & (~part_df['interactions'].str.contains('Qa')) & (~part_df['interactions'].str.contains('Na'))
                    & (~part_df['interactions'].str.contains('C1')) & (~part_df['interactions'].str.contains('C3'))
                    & (~part_df['interactions'].str.contains('POL')) & (~part_df['interactions'].str.contains('PQd'))]

    return cl, pg, qa, na, c1, c3, water, ion, other


def load_data(dir_path, two_body, three_body, principal_component):  # one_body,
    fig = plt.figure(constrained_layout=True, figsize=(16, 9), dpi=150)
    gs = mpl.gridspec.GridSpec(nrows=2, ncols=2, figure=fig, left=0.02, bottom=0.02, right=0.98, top=None, wspace=None,
                               hspace=None, width_ratios=None, height_ratios=None)
    df = process_data(dir_path, two_body, three_body)
    groups = df.groupby(level='body', sort=False)
    increment = np.array([0, 1])
    for idx, (interaction, group) in enumerate(groups):
        pc = group[principal_component]
        pos = pc[pc['weights'] > 0.0]
        neg = pc[pc['weights'] < 0.0]
        if len(pos.index) < 21:
            pos = pos
        else:
            pos = pos.nlargest(21, ['weights'])
        if len(neg.index) < 21:
            neg = neg.iloc[neg.weights.abs().argsort()]
        else:
            neg = neg.iloc[neg.weights.abs().argsort()].tail(21)

        pos_cl, pos_pg, pos_qa, pos_na, pos_c1, pos_c3, pos_water, pos_ion, pos_other = separate_members(pos)
        neg_cl, neg_pg, neg_qa, neg_na, neg_c1, neg_c3, neg_water, neg_ion, neg_other = separate_members(neg)

        # print(idx + increment[0], idx + increment[1])
        # plot_interactions(pos_cl, pos_pg, neg_cl, neg_pg, axes[idx + increment[0]], axes[idx + increment[1]])
        plot_interactions(pos_cl, pos_pg, neg_cl, neg_pg, fig.add_subplot(gs[0, idx]), fig.add_subplot(gs[1, idx]))
        # increment = increment + 1

    # plt.tight_layout()
    fig.suptitle(principal_component)
    path = dir_path / f'{principal_component}_ranked_weights_headgroups.pdf'
    plt.savefig(path)
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Sort interactions, save as table')
    parser.add_argument('-dir', '--directory', type=Path, required=True, help='Directory containing interaction '
                                                                              'dataframes. Plot will be saved there.')
    # parser.add_argument('-one', '--one_body', type=Path, required=True, help='Df with one-body interactions.')
    parser.add_argument('-two', '--two_body', type=Path, required=True, help='Df with two-body interactions.')
    parser.add_argument('-three', '--three_body', type=Path, required=True, help='Df with three-body interactions.')
    parser.add_argument('-pc', '--principal_component', type=str, required=True, help='Column name for weights to '
                                                                                      'analyze.')

    args = parser.parse_args()

    load_data(args.directory, args.two_body, args.three_body, args.principal_component)  # args.one_body,
