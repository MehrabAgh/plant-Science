import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from bioinfokit.visuz import cluster
import sys

def get_indices_df(genotype_code, Yp, Ys, RC, TOLL, MP, GMP, HM, SSI, STI, YI, YSI, RSI):
    return pd.DataFrame({        
                    'Genotype Code': genotype_code,
                    'Yp':Yp,
                    'Ys':Ys,
                    'RC':RC,
                    'TOLL':TOLL,
                    'MP':MP,
                    'GMP':GMP,
                    'HM':HM,
                    'SSI':SSI,
                    'STI':STI,
                    'YI':YI,
                    'YSI':YSI,
                    'RSI':RSI
            })
def get_ranks_df(genotype_code, Yp, Ys, TOLL, MP, GMP, HM, SSI, STI, YI, YSI, RSI):
    Yp_ranks = Yp.rank(ascending=False, method='min').astype(int)
    Ys_ranks = Ys.rank(ascending=False, method='min').astype(int)
    TOLL_ranks = TOLL.rank(ascending=True, method='min').astype(int)
    MP_ranks = MP.rank(ascending=False, method='min').astype(int)
    GMP_ranks = GMP.rank(ascending=False, method='min').astype(int)
    HM_ranks = HM.rank(ascending=False, method='min').astype(int)
    SSI_ranks = SSI.rank(ascending=True, method='min').astype(int)
    STI_ranks = STI.rank(ascending=False, method='min').astype(int)
    YI_ranks = YI.rank(ascending=False, method='min').astype(int)
    YSI_ranks = YSI.rank(ascending=False, method='min').astype(int)
    RSI_ranks = RSI.rank(ascending=False, method='min').astype(int)

    return pd.DataFrame({
                            'Genotype Code': genotype_code,
                            'Yp':Yp_ranks,
                            'Ys':Ys_ranks,
                            'TOLL':TOLL_ranks,
                            'MP':MP_ranks,
                            'GMP':GMP_ranks,
                            'HM':HM_ranks,
                            'SSI':SSI_ranks,
                            'STI': STI_ranks,
                            'YI':YI_ranks,
                            'YSI':YSI_ranks,
                            'RSI':RSI_ranks,
                        })
    numeric_only_ranks_df = ranks_df.select_dtypes(include=np.number)
    SR = numeric_only_ranks_df.sum(axis=1)
    AR = SR / len(numeric_only_ranks_df)
    SD = numeric_only_ranks_df.std(axis=1)

    ranks_df['SR'] = SR
    ranks_df['AR'] = AR
    ranks_df['SD'] = SD

    return ranks_df
def generate_3d_plot_html_file(df, z, x, y, path='public/data_results/fig.html'):
    fig = px.scatter_3d(df, x=x, y=y, z=z, color=df.columns[0])
    plotly.offline.plot(fig, filename=path, auto_open=False)
def compute_histogram_bins(data, desired_bin_size):
    min_val = np.min(data)
    max_val = np.max(data)
    min_boundary = -1.0 * (min_val % desired_bin_size - min_val)
    max_boundary = max_val - max_val % desired_bin_size + desired_bin_size
    n_bins = int((max_boundary - min_boundary) / desired_bin_size) + 1
    bins = np.linspace(min_boundary, max_boundary, n_bins)
    return bins
def generate_relative_frequency_bar_graph_image(df, feature_name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    number_of_rows = df.shape[0]
    bin_size = math.floor(1 + 3.322 * math.log(number_of_rows, 10))
    plt.xlabel(feature_name)
    plt.ylabel('Relative Frequency')
    bins = compute_histogram_bins(data=df[feature_name], desired_bin_size=bin_size)
    ax.hist(df[feature_name], edgecolor='black', weights=np.ones_like(df[feature_name]) / number_of_rows, bins=7)
    plt.savefig('public/data_results/relative_frequency.png')
def generate_correlations_heatmaps_images(df):
    fig, axs = plt.subplots(2, 1, figsize=(30, 30))
    df = df.drop([df.columns[0], 'RC'], axis=1)
    Pearsons_heatmap = df.corr(method='pearson')
    spearmans_heatmap = df.corr(method='spearman')
    plot = sns.heatmap(Pearsons_heatmap, annot=True, square=True, ax=axs[0])
    axs[0].set_title("Pearson's Correlation Heatmap")
    plot = sns.heatmap(spearmans_heatmap, annot=True, square=True, ax=axs[1])
    axs[1].set_title("Spearman's Correlation Heatmap")
    plot.get_figure().savefig('public/data_results/correlations_heatmaps.png', bbox_inches='tight') 
def generate_pca_plot_image(df):
    X = df.select_dtypes(include=np.number)
    X_st =  StandardScaler().fit_transform(X)
    pca_out = PCA().fit(X_st)
    loadings = pca_out.components_
    pca_out.explained_variance_
    pca_scores = PCA().fit_transform(X_st)
    cluster.biplot(cscore=pca_scores, loadings=loadings, labels=X.columns.values, var1=round(pca_out.explained_variance_ratio_[0]*100, 2),
        var2=round(pca_out.explained_variance_ratio_[1]*100, 2))    
def Calculate ():   
    df = pd.read_excel("uploads/"+sys.argv[1])
    num_rows = len(df.index)
    if (num_rows == 3):
        genotype_code = df[df.columns[0]]
        Yp, Ys = df.Yp, df.Ys
        Yp_mean, Ys_mean = Yp.mean(), Ys.mean()
        RC = (Yp - Ys) / Yp * 100
        TOLL = Yp - Ys
        MP = (Yp + Ys) / 2
        GMP = np.sqrt(Ys * Yp)
        HM = 2 * Ys * Yp / (Ys + Yp)
        SSI = (1 - Ys / Yp) / (1 - Ys_mean / Yp_mean)
        STI = Ys * Yp / Yp_mean ** 2
        YI = Ys / Ys_mean
        YSI = Ys / Yp
        RSI = (Ys / Yp) / (Ys_mean / Yp_mean)
        indices_df = get_indices_df(genotype_code, Yp, Ys, RC, TOLL, MP, GMP, HM, SSI, STI, YI, YSI, RSI)
        ranks_df = get_ranks_df(genotype_code, Yp, Ys, TOLL, MP, GMP, HM, SSI, STI, YI, YSI, RSI)

        generate_relative_frequency_bar_graph_image(indices_df,sys.argv[2])
        generate_correlations_heatmaps_images(indices_df)
        generate_3d_plot_html_file(indices_df, x='Yp', y='Ys', z=sys.argv[2])
        generate_pca_plot_image(indices_df)

        data_df= {
          "indices_df": indices_df,
          "ranks_df": ranks_df
        }
        return data_df
    else : 
        return False
    
from subprocess import Popen, PIPE
import json

  
  

process = Popen(['node', 'index.js', { "name" : "ss"} ], stdout=PIPE)
  
stdout = process.communicate()[0]

result_data = json.loads(stdout)
array_sum = result_data['sum']
print('Sum of array from Node.js process =',array_sum)
