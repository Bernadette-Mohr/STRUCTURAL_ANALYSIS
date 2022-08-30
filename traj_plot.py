import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_pickle('/media/bmohr/Backup/STRUCTURAL_ANALYSIS/MDAnalysis_stats_all.pickle')
df['frame'] = df['frame'].astype(int)
# print(df)

raw = df[(df['state'] == 'RAW')]
corr = df[(df['state'] == 'CORR')]
sel = df[(df['state'] == 'SEL')]

# df_all = raw.merge(corr, on=['round', 'molecule', 'frame'], how='left', indicator=True)
# only_in_raw = df_all[df_all['_merge'] == 'left_only']
# print(only_in_raw)

# sel.hist(column='frame')
# print(sel['frame'].max())
# plt.show()
# popg = sel[sel['molecule'].str.contains('POPG')]
# cdl2 = sel[sel['molecule'].str.contains('CDL2')]


def flag_mols_with_dupes():
    rounds = sorted(set(sel['round'].tolist()))
    lipids = sorted(set(name.split('-')[0] for name in sel['molecule'].tolist()))
    iterables = [rounds, lipids]
    columns = pd.MultiIndex.from_product(iterables, names=['round', 'lipid'])
    problems = pd.DataFrame(columns=columns)
    for rnd in rounds:
        print(rnd)
        rnd_df = sel[sel['round'] == rnd]
        for lipid in lipids:
            problems[(rnd, lipid)] = pd.Series(sorted(set(name.split('-')[1] for name in rnd_df['molecule'].tolist()
                                                          if lipid in name)))
    print(problems)
    problems.to_pickle('/media/bmohr/Backup/STRUCTURAL_ANALYSIS/problematic_molecules.pickle')


def plot_beadtypes():
    fig, ax = plt.subplots(dpi=150)
    ax2 = ax.twiny()
    unique = corr['name_unique']
    dupe = corr['name_dupe']
    unique.value_counts().plot(kind='bar', color='blue', alpha=0.3, width=1, ax=ax)
    # ax.set_xticklabels(ax.get_xticks(), rotation=0)
    dupe.value_counts().plot(kind='bar', color='orange', alpha=0.3, width=1, ax=ax2)
    # ax2.set_xticklabels(ax2.get_xticks(), rotation=0)
    # corr.hist(column='name_dupe', ax=ax)
    plt.yscale('log')
    # plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


plot_beadtypes()
