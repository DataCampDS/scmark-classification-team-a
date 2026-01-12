import numpy as np
from scipy import sparse

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_extraction.text import TfidfTransformer


class Classifier:
    def __init__(
        self,
        min_nnz=3,
        pre_k=8000,
        use_tfidf=True,
        n_components=300,
        C=10.0,
        random_state=42,
    ):
        self.min_nnz = min_nnz
        self.pre_k = pre_k
        self.use_tfidf = use_tfidf
        self.n_components = n_components
        self.C = C
        self.random_state = random_state

        self.keep_ = None
        self.pre_idx_ = None
        self.model_ = None

    @staticmethod
    def _to_csr(X):
        if sparse.issparse(X) and not sparse.isspmatrix_csr(X):
            return X.tocsr()
        return X

    @staticmethod
    def _prefilter_by_nnz(X, min_nnz):
        nnz = np.asarray((X != 0).sum(axis=0)).ravel()
        keep = nnz >= min_nnz
        return X[:, keep], keep

    @staticmethod
    def _topk_dispersion(X, k):
        # dispersion on raw counts 
        mean = np.asarray(X.mean(axis=0)).ravel()
        mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
        var = mean_sq - mean**2
        var[var < 0] = 0.0
        disp = var / (mean + 1e-6)

        k = min(k, X.shape[1])
        idx = np.argpartition(disp, -k)[-k:]
        idx = idx[np.argsort(disp[idx])[::-1]]
        return idx

    def fit(self, X, y):
        if not sparse.issparse(X):
            X = sparse.csr_matrix(X)
        else:
            X = self._to_csr(X)

        # remove rare genes
        X, keep = self._prefilter_by_nnz(X, self.min_nnz)
        self.keep_ = keep

        # top-k by dispersion to cap dimension
        idx = self._topk_dispersion(X, self.pre_k)
        self.pre_idx_ = idx
        Xp = X[:, idx]

        # TF-IDF  + SVD + LogReg
        steps = []
        if self.use_tfidf:
            steps.append(("tfidf", TfidfTransformer(sublinear_tf=True)))
        else:
            pass

        n_comp = min(self.n_components, max(2, Xp.shape[1] - 1))
        steps += [
            ("svd", TruncatedSVD(n_components=n_comp, random_state=self.random_state)),
            ("clf", LogisticRegression(
                penalty="l2",
                C=self.C,
                max_iter=4000,
                tol=1e-3,
                class_weight="balanced",
                n_jobs=-1,
                random_state=self.random_state
            ))
        ]

        self.model_ = Pipeline(steps)

        if self.use_tfidf:
            self.model_.fit(Xp, y)
        else:
            Xp = normalize(Xp, norm="l2", axis=1)
            self.model_.fit(Xp, y)

        return self

    def predict_proba(self, X):
        if not sparse.issparse(X):
            X = sparse.csr_matrix(X)
        else:
            X = self._to_csr(X)

        X = X[:, self.keep_]
        Xp = X[:, self.pre_idx_]

        if self.use_tfidf:
            return self.model_.predict_proba(Xp)
        else:
            Xp = normalize(Xp, norm="l2", axis=1)
            return self.model_.predict_proba(Xp)
