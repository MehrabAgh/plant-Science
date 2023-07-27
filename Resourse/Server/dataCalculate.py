import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def GMP(Ys, Yp, YsBar, YpBar):
    return (pow(Ys*Yp , 0.5))
def RC(Ys, Yp, YsBar, YpBar):
    return (((Yp - Ys) / Yp) * 100)
def TOL(Ys, Yp, YsBar, YpBar):
    return (Yp - Ys)
def MP(Ys, Yp, YsBar, YpBar):
    return ((Yp + Ys) / 2)
def HM(Ys, Yp, YsBar, YpBar):
    return (2 * (Ys * Yp) / (Ys + Yp))
def SSI(Ys, Yp, YsBar, YpBar):
    return ((1 - Ys / Yp) / (1 - YsBar / YpBar))
def STI(Ys, Yp, YsBar, YpBar):
    return ((Ys * Yp) / (YpBar ** 2))
def YI(Ys, Yp, YsBar, YpBar):
    return (Ys / YsBar)
def YSI(Ys, Yp, YsBar, YpBar):
    return (Ys / Yp)
def RSI(Ys, Yp, YsBar, YpBar):
    return ((Ys / Yp) / (YsBar / YpBar))
def setNumberPrecisionToMatchExcel(num):
    return round(num, 15)

def getranks_df(df_orig):
    descendings = [2,3,6,7,8,10,11,12,13]
    for col in descendings:
        df_orig[col] = df_orig[col] * (-1)
    
    df = df_orig.iloc[:, [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
    ranks = df.apply(lambda x: pd.Series(x).rank(method="min"))
    
    SR = ranks.sum(axis=1)
    SR = pd.DataFrame(SR)
    SR.columns = ["SR"]
    
    AR = SR / len(ranks.columns)
    AR.columns = ["AR"]
    
    STD = ranks.std(axis=1)
    STD = pd.DataFrame(STD)
    STD.columns = ["Std."]
    
    ranks = pd.concat([df_orig.iloc[:,0], ranks], axis=1)
    ranks = pd.concat([ranks, SR], axis=1)
    ranks = pd.concat([ranks, AR], axis=1)
    ranks = pd.concat([ranks, STD], axis=1)
    
    return(ranks)
def getSignChangingVec(Matr):
    mat = np.apply_along_axis(setNumberPrecisionToMatchExcel, axis=0, arr=Matr)
    signChangingFlags = []
    for i in range(mat.shape[1]):
        x = mat[:, i]
        max = None
        maxInd = 0
        for j in range(len(x)):
            if max is None or abs(x[j]) - max > (5e-16) or (abs(abs(x[j]) - max) < (5e-16) and x[j] < 0):
                max = abs(x[j])
                maxInd = j
        colsign = np.sign(x[maxInd])
        signChangingFlags.append(colsign)
    return signChangingFlags
def getpcLabels(count):
    labels = []
    for i in range(count):
        labels.append("PC" + str(i))
    return labels
def getpca(df, iscov):
    genonames = df.index
    factornames = df.columns
    eigenvals = None
    eigenvecs = None
    loadings = None
    contributions = None
    scores = None
    scoresignchangevec = None
    if not iscov:
        eig = np.linalg.eig(np.corrcoef(df, rowvar=False))
        pca = PCA(n_components=df.shape[1], svd_solver='full')
        pca.fit_transform(df)
        scoresignchangevec = np.sign(pca.components_)[np.argmax(np.abs(pca.components_), axis=1), range(pca.components_.shape[1])]
        scores = pca.transform(df) * scoresignchangevec
    else:
        eig = np.linalg.eig(np.cov(df, rowvar=False))
        pca = PCA(n_components=df.shape[1], svd_solver='full')
        pca.fit_transform(df)
        scores = pca.inverse_transform(df)
        scoresignchangevec = np.sign(pca.components_)[np.argmax(np.abs(pca.components_), axis=1), range(pca.components_.shape[1])]
        scores = scores * scoresignchangevec
    eigenvals = eig[0]
    eigenvecs = eig[1]
    loadingCalcEigenvals = eigenvals.copy()
    loadingCalcEigenvals[loadingCalcEigenvals < 0] = np.nan
    loadings = eigenvecs * np.sqrt(loadingCalcEigenvals)
    loadings = loadings[:, ~np.isnan(loadings).any(axis=0)]
    eigvecs_p2 = eigenvecs ** 2
    contributions = (eigvecs_p2) / np.sum(eigvecs_p2, axis=0) * 100
    signChangingFlags = np.sign(loadings)
    eigenvecs = eigenvecs[:, :signChangingFlags.shape[1]]
    scores = scores[:, :min(scores.shape[1], signChangingFlags.shape[1])]
    eigenvals = pd.DataFrame(eigenvals).T
    importancedf = pd.DataFrame({'Standard deviation': np.sqrt(pca.explained_variance_), 'Proportion of Variance': pca.explained_variance_ratio_}).T
    eigenvals = pd.concat([eigenvals.iloc[:, :min(importancedf.shape[1], eigenvals.shape[1])], importancedf.iloc[:, :min(importancedf.shape[1], eigenvals.shape[1])]])
    eigenvals.columns = ['PC' + str(i) for i in range(1, eigenvals.shape[1] + 1)]
    eigenvals.index = ['Eigenvalue', 'Variability (%)', 'Cumulative %']
    eigenvecs = pd.DataFrame(eigenvecs)
    eigenvecs.columns = ['PC' + str(i) for i in range(1, eigenvecs.shape[1] + 1)]
    eigenvecs.index = factornames[:eigenvecs.shape[0]]
    loadings = pd.DataFrame(loadings)
    loadings.columns = ['PC' + str(i) for i in range(1, loadings.shape[1] + 1)]
    loadings.index = factornames[:loadings.shape[0]]
    contributions = pd.DataFrame(contributions)
    contributions.columns = ['PC' + str(i) for i in range(1, contributions.shape[1] + 1)]
    contributions.index = factornames[:contributions.shape[0]]
    if not iscov:
        pca = PCA(n_components=df.shape[1], svd_solver='full')
        pca.fit_transform(df)
    else:
        pca = PCA(n_components=df.shape[1], svd_solver='full')
        pca.fit(df)
    pca.components_ = pca.components_ * np.sign(pca.components_[:, ~np.all(np.isnan(pca.components_), axis=0)])
    pca.transform(df)[:, ~np.all(np.isnan(pca.transform(df)), axis=0)] = pca.transform(df)[:, ~np.all(np.isnan(pca.transform(df)), axis=0)] * np.sign(pca.transform(df)[:, ~np.all(np.isnan(pca.transform(df)), axis=0)])
    return {'eigenvals': eigenvals, 'eigenvecs': eigenvecs * signChangingFlags, 'loadings': loadings * signChangingFlags, 'contributions': contributions.iloc[:, :signChangingFlags.shape[1]], 'scores': scores, 'pca_obj': pca}

def Calculate(table_original): 
    table = table_original.iloc[:, 1:]     
           
    Yp = table.iloc[:, 0]
    Ys = table.iloc[:, 1]  
    YpBar = np.mean(Yp)
    YsBar = np.mean(Ys)       

    def runFunc(func): 
        return func(Ys, Yp, YsBar, YpBar) 
    
    stats_df = pd.DataFrame({
        "Species": table_original.iloc[:, 0],
        "Yp": table_original.iloc[:, 1],
        "Ys": table_original.iloc[:, 2],
        "RC": runFunc(RC),
        "TOL": runFunc(TOL),
        "MP": runFunc(MP),
        "GMP": runFunc(GMP),
        "HM": runFunc(HM),
        "SSI": runFunc(SSI),
        "STI": runFunc(STI),
        "YI": runFunc(YI),
        "YSI": runFunc(YSI),
        "RSI": runFunc(RSI)
    })   
    ranks_df = stats_df.rank()
    
    pcadf = stats_df.iloc[:, 3:].drop(columns=["TOL"]) 
    pcadf.index = stats_df.iloc[:, 0]            
    correlation_based_pca = getpca(pcadf, False) 
    covariance_based_pca = getpca(pcadf, True)    
    if 'scores' in correlation_based_pca:
        correlation_based_pca['scores'] = stats_df.iloc[:, 0]
    if 'scores' in covariance_based_pca:
        covariance_based_pca['scores'] = stats_df.iloc[:, 0]   

    ranks_df.index = ranks_df.iloc[:, 0] 
    ranks_df = ranks_df.iloc[:, 0:] 
     
    correlations = {
        "pearson":np.squeeze(np.corrcoef(stats_df.iloc[:, 2:].values.T)),
        "spearman": np.corrcoef(ranks_df.iloc[:, 1:].values.T)
    }
     
    output = {
        "indices": stats_df,
        "ranks": ranks_df,
        "correlations": correlations,
        "pca": {
            "correlation_based": correlation_based_pca,
            "covariance_based": covariance_based_pca
        }
    } 
    return output