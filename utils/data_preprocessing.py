import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(descriptor_path, label_path, compo=100):
    X = np.load(descriptor_path, allow_pickle=True)
    y = np.load(label_path, allow_pickle=True)

    X = X[:, ~np.all(np.isnan(X), axis=0)]

    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    pca = PCA(n_components=compo)
    X = pca.fit_transform(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y
