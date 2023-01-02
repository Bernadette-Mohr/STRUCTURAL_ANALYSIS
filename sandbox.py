from plotly.offline import plot
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

iris = load_iris()
pca = PCA(n_components=4).fit(iris.data)
X_reduced = pca.transform(iris.data)
trace1 = go.Scatter3d(x=X_reduced[:, 0], y=X_reduced[:, 1], z=X_reduced[:, 2], mode='markers',
                      marker=dict(size=12, color=iris.target, opacity=1))

dc_1 = go.Scatter3d(x=[0, pca.components_.T[0][0]],
                    y=[0, pca.components_.T[0][1]],
                    z=[0, pca.components_.T[0][2]],
                    marker=dict(size=1, color="rgb(84,48,5)"),
                    line=dict(color="red", width=6),
                    name="Var1")
dc_2 = go.Scatter3d(x=[0, pca.components_.T[1][0]],
                    y=[0, pca.components_.T[1][1]],
                    z=[0, pca.components_.T[1][2]],
                    marker=dict(size=1, color="rgb(84,48,5)"),
                    line=dict(color="green", width=6),
                    name="Var2")
dc_3 = go.Scatter3d(x=[0, pca.components_.T[2][0]],
                    y=[0, pca.components_.T[2][1]],
                    z=[0, pca.components_.T[2][2]],
                    marker=dict(size=1,
                                color="rgb(84,48,5)"),
                    line=dict(color="blue", width=6),
                    name="Var3")
dc_4 = go.Scatter3d(x=[0, pca.components_.T[3][0]],
                    y=[0, pca.components_.T[3][1]],
                    z=[0, pca.components_.T[3][2]],
                    marker=dict(size=1, color="rgb(84,48,5)"),
                    line=dict(color="yellow", width=6),
                    name="Var4")


data = [trace1, dc_1, dc_2, dc_3, dc_4]
layout = go.Layout(
    xaxis=dict(
        title='PC1',
        titlefont=dict(
           family='Courier New, monospace',
           size=18,
           color='#7f7f7f'
        )
    )
)
fig = go.Figure(data=data, layout=layout)
# fig.show()
plot(fig, filename='3d-scatter-tupac-with-mac.html')

# def get_largest_correlations(weight_corr):
#     path = '/home/bernadette/Documents/STRUCTURAL_ANALYSIS'
#     weight_corr = weight_corr.T
#     weight_corr = weight_corr.corr()
#     corr_stack = weight_corr.stack()
#     corr_stack = corr_stack[corr_stack.index.get_level_values(0) != corr_stack.index.get_level_values(1)]
#     sim_idx = list()
#     for first, second in corr_stack.index.tolist():
#         _first = first.split('-')
#         _second = second.split('-')
#         if len(_first) < len(_second):
#             for element in _first:
#                 if element in _second:
#                     _second.remove(element)
#                 else:
#                     break
#                 if len(_second) < len(_first):
#                     if (first, second) not in sim_idx and (second, first) not in sim_idx:
#                         sim_idx.append((first, second))
#         elif len(_first) > len(_second):
#             for element in _second:
#                 if element in _first:
#                     _first.remove(element)
#                 else:
#                     break
#                 if len(_first) < len(_second):
#                     if (first, second) not in sim_idx and (second, first) not in sim_idx:
#                         sim_idx.append((first, second))
#     ident_idx = list()
#     for first, second in corr_stack.index.tolist():
#         _first = first.split('-')
#         _second = second.split('-')
#         if len(_first) == len(_second):
#             if sorted(_first) == sorted(_second):
#                 if (first, second) not in ident_idx and (second, first) not in ident_idx:
#                     ident_idx.append((first, second))
#     similar = corr_stack.loc[sim_idx]
#     similar = similar.loc[similar.abs().nlargest(len(similar.index)).index]
#     sim1, sim2 = map(list, zip(*similar.abs().nlargest(20).index.tolist()))
#     similar = weight_corr.loc[sim1, sim2]
#     # sim_fig = plt.figure(figsize=(9, 8), dpi=150)
#     min_ = min(similar.min())
#     max_ = max(similar.max())
#     if min_ < 0:
#         center = 0.0
#     else:
#         center = ((max_ - min_) / 2) + min_
#     # divnorm = colors.TwoSlopeNorm(vmin=min_, vcenter=center, vmax=max_)
#     # sim_ax = sns.heatmap(similar, annot=True, square=True, fmt=".1f", cmap='bwr', norm=divnorm)  #
#     # plt.yticks(rotation=0)
#     # plt.xticks(rotation=45)
#     # plt.tight_layout()
#     # plt.savefig(f'{path}/correlation_two-body_three_body.pdf')
#     identical = corr_stack.loc[ident_idx]
#     identical = corr_stack.loc[identical.abs().nlargest(len(identical.index)).index]
#     ident1, ident2 = map(list, zip(*identical.abs().nlargest(20).index.tolist()))
#     identical = weight_corr.loc[ident1, ident2]
#     # ident_fig = plt.figure(figsize=(9, 8), dpi=150)
#     # min_ = min(identical.min())
#     # max_ = max(identical.max())
#     # if min_ < 0:
#     #     center = 0.0
#     # else:
#     #     center = ((max_ - min_) / 2) + min_
#     # divnorm = colors.TwoSlopeNorm(vmin=min_, vcenter=center, vmax=max_)
#     # ident_ax = sns.heatmap(identical, annot=True, square=True, fmt=".1f",
#     #                        cmap='bwr', norm=divnorm)
#     # plt.tight_layout()
#     # plt.savefig(f'{path}/correlation_three-body_three_body.pdf')
#     sim_idx.extend(ident_idx)
#     try:
#         different = pd.read_pickle(f'{path}/different_interactions_correlation_matrix.pickle')
#     except FileNotFoundError:
#         all_pairs = corr_stack.index.tolist()
#         print(len(all_pairs))
#         keep_idx = list()
#         for idx in tqdm(all_pairs):
#             first = sorted(idx[0].split('-'))
#             second = sorted(idx[1].split('-'))
#             if idx not in sim_idx and (idx[1], idx[0]) not in sim_idx:
#                 keep_idx.append(idx)
#             else:
#                 try:
#                     all_pairs.remove(idx)
#                     all_pairs.remove((idx[1], idx[0]))
#                 except ValueError:
#                     pass
#             last_elem = sim_idx[-1]
#             for old_idx in sim_idx:
#                 old_first = sorted(old_idx[0].split('-'))
#                 old_second = sorted(old_idx[1].split('-'))
#                 if first == old_first and second == old_second:
#                     try:
#                         all_pairs.remove(idx)
#                         all_pairs.remove((idx[1], idx[0]))
#                     except ValueError:
#                         pass
#                 elif first == old_second and second == old_first:
#                     try:
#                         all_pairs.remove(idx)
#                         all_pairs.remove((idx[1], idx[0]))
#                     except ValueError:
#                         pass
#                     continue
#                 else:
#                     if old_idx == last_elem:
#                         keep_idx.append(idx)
#                         try:
#                             all_pairs.remove(idx)
#                             all_pairs.remove((idx[1], idx[0]))
#                         except ValueError:
#                             pass
#         print(len(all_pairs))
#         print(keep_idx)
#         for first, second in tqdm(all_pairs):
#             check_new = check_if_present(first, second, keep_idx)
#             if check_new:
#                 keep_idx.append((first, second))
#         print(keep_idx)
#         different = corr_stack.loc[keep_idx]
#         different = different.loc[different.abs().nlargest(len(keep_idx)).index]
#         diff1, diff2 = map(list, zip(*different.abs().nlargest(len(keep_idx)).index.tolist()))
#         save_ = weight_corr.loc[diff1, diff2]
#         save_.to_pickle(f'{path}/different_interactions_correlation_matrix_all.pickle')
#         del save_
#         diff1, diff2 = map(list, zip(*different.abs().nlargest(20).index.tolist()))
#         different = weight_corr.loc[diff1, diff2]
#         # different.to_pickle(f'{path}/different_interactions_correlation_matrix.pickle')
#     # diff_fig = plt.figure(figsize=(9, 8), dpi=150)
#     # min_ = min(different.min())
#     # max_ = max(different.max())
#     # if min_ < 0:
#     #     center = 0.0
#     # else:
#     #     center = ((max_ - min_) / 2) + min_
#     # print(min_, center, max_)
#     # divnorm = colors.TwoSlopeNorm(vmin=min_, vcenter=center, vmax=max_)
#     # diff_ax = sns.heatmap(different, annot=True, square=True, fmt=".1f", cmap='bwr', norm=divnorm)  #
#     # plt.yticks(rotation=0)
#     # plt.xticks(rotation=45)
#     # plt.tight_layout()
#     # plt.savefig(f'{path}/correlation_differences.pdf')
