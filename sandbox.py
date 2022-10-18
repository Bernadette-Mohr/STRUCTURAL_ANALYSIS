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
