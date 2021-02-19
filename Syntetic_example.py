from plotly.subplots import make_subplots
import numpy as np
import plotly.graph_objects as go
import plotly
import tqdm
from sklearn.decomposition import PCA

d_set = [1000 , 100]
n_set = [1000 , 100]
total_exper = 100

dict_results = {}

for d in tqdm.tqdm(d_set):
    dict_results[d] = {}
    for n in n_set:
        collect_results = []
        for _ in range(total_exper):
            # parameters of synthetic data
            mean1 = np.zeros(d)
            mean1[0] = 0.5
            cov1 = np.diag(np.ones(d)) * 0.05
            mean2 = np.zeros(d)
            mean2[0] = 0.1
            # generation of synthetic data
            x1 = np.random.multivariate_normal(mean1, cov1, int(n / 2))
            x2 = np.random.multivariate_normal(mean2, cov1, int(n / 2))
            XT = np.concatenate([x1, x2])
            # application of PCA
            pcal = PCA(n_components=np.min(XT.shape), svd_solver='full')
            pcal.fit(XT.T)
            collect_results.append(pcal.components_[0])

        dict_results[d][n] = np.mean(collect_results, axis=0)

cols = plotly.colors.DEFAULT_PLOTLY_COLORS
fig = make_subplots(rows=1, cols=3,
                    subplot_titles=(r' $ d=1000; \ n=1000;   $', r'$d=1000; \ n=100; $', r'$ d=100; \ n=1000;  $'))
error_type = 0

fig.add_trace(go.Scatter(x=np.arange(1000), y=dict_results[1000][1000], mode='lines',
                         line=dict(width=2, color=cols[2]),
                         name=r'$U_0 $', showlegend=True, ),
              row=1, col=1, )
fig.add_trace(go.Scatter(x=np.arange(100), y=dict_results[1000][100], mode='lines',
                         line=dict(width=2, color=cols[1]),
                         name=r'$U_0 $', showlegend=True, ),
              row=1, col=2, )

fig.add_trace(go.Scatter(x=np.arange(1000), y=dict_results[100][1000], mode='lines',
                         line=dict(width=2, color=cols[3]),
                         name=r'$U_0 $', showlegend=True, ),
              row=1, col=3, )

fig.update_xaxes(title_text=r'$ n $')
fig.update_yaxes(title_text=r'$ U_0 $', row=1, col=1, )

fig.update_layout(height=600, width=1400, legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
), font=dict(
    family="Courier New, monospace",
    size=28
))
fig.show()
