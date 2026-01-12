    import numpy as np
    from scipy import sparse
     
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import TruncatedSVD
    from sklearn.linear_model import LogisticRegression
     
     
    class Classifier:
        def __init__(
            self,
            n_bins=20,
            pre_k=6000,
            min_nnz=10,
            scale=1e4,
            C=10.0,
            l1_ratio=0.5,
            n_components=300,
            use_cell_qc=True,
            class_weight="balanced",
            random_state=42,
        ):
            self.n_bins = n_bins
            self.pre_k = pre_k
            self.min_nnz = min_nnz
            self.scale = scale
            self.C = C
            self.l1_ratio = l1_ratio
            self.n_components = n_components
            self.use_cell_qc = use_cell_qc
            self.class_weight = class_weight
            self.random_state = random_state
     
            self.keep_ = None
            self.pre_idx_ = None
            self.model_ = None
     
        @staticmethod
        def _to_csr(X):
            return X.tocsr() if sparse.issparse(X) and not sparse.isspmatrix_csr(X) else X
     
        @staticmethod
        def _cp10k_log1p_csr(X, scale=1e4):
            X = X.tocsr().copy()
            lib = np.asarray(X.sum(axis=1)).ravel()
            lib[lib == 0] = 1.0
            X = X.multiply((scale / lib)[:, None]).tocsr()
            X.data = np.log1p(X.data)
            return X, lib
     
        def _hvg_bins(self, X):
            # mean/var on CSR
            mean = np.asarray(X.mean(axis=0)).ravel()
            mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
            var = mean_sq - mean**2
            var[var < 0] = 0.0
     
            disp = var / (mean + 1e-6)
     
            # bin genes by mean quantiles
            qs = np.linspace(0, 1, self.n_bins + 1)
            edges = np.quantile(mean, qs)
            edges[0] -= 1e-9
            edges[-1] += 1e-9
     
            scores = np.zeros_like(disp)
            for b in range(self.n_bins):
                m = (mean >= edges[b]) & (mean < edges[b + 1])
                if m.sum() < 10:
                    scores[m] = -np.inf
                    continue
                d = disp[m]
                mu = d.mean()
                sd = d.std() + 1e-12
                scores[m] = (d - mu) / sd  # z-score within bin
     
            k = min(self.pre_k, X.shape[1])
            idx = np.argpartition(scores, -k)[-k:]
            idx = idx[np.argsort(scores[idx])[::-1]]
            return idx
     
        def fit(self, X, y):
            if not sparse.issparse(X):
                X = sparse.csr_matrix(X)
            else:
                X = self._to_csr(X)
     
            # QC features computed on raw counts (before normalization)
            lib_raw = np.asarray(X.sum(axis=1)).ravel()
            lib_raw[lib_raw == 0] = 1.0
            pct_zeros = 1.0 - (np.asarray((X != 0).sum(axis=1)).ravel() / X.shape[1])
     
            # CP10k + log1p
            X, _ = self._cp10k_log1p_csr(X, scale=self.scale)
     
            # remove rare genes
            nnz = np.asarray((X != 0).sum(axis=0)).ravel()
            keep = nnz >= self.min_nnz
            self.keep_ = keep
            X = X[:, keep]
     
            # HVG bins selection
            pre_idx = self._hvg_bins(X)
            self.pre_idx_ = pre_idx
            Xp = X[:, pre_idx]
     
            # model: SVD + ElasticNet
            n_comp = min(self.n_components, max(2, Xp.shape[1] - 1))
            base = Pipeline([
                ("scaler", StandardScaler(with_mean=False)),
                ("svd", TruncatedSVD(n_components=n_comp, random_state=self.random_state)),
                ("clf", LogisticRegression(
                    solver="saga",
                    penalty="elasticnet",
                    l1_ratio=self.l1_ratio,
                    C=self.C,
                    max_iter=4000,
                    tol=1e-3,
                    class_weight=self.class_weight,
                    n_jobs=-1,
                    random_state=self.random_state
                ))
            ])
     
            if self.use_cell_qc:
                
                Z = Pipeline(base.steps[:-1]).fit_transform(Xp)
                qc = np.vstack([np.log1p(lib_raw), pct_zeros]).T
                Z2 = np.hstack([Z, qc])
     
                clf = base.named_steps["clf"]
                clf.fit(Z2, y)
     
                self.model_ = ("with_qc", Pipeline(base.steps[:-1]), clf)
            else:
                self.model_ = ("no_qc", base)
                base.fit(Xp, y)
     
            return self
     
        def predict_proba(self, X):
            if not sparse.issparse(X):
                X = sparse.csr_matrix(X)
            else:
                X = self._to_csr(X)
     
            lib_raw = np.asarray(X.sum(axis=1)).ravel()
            lib_raw[lib_raw == 0] = 1.0
            pct_zeros = 1.0 - (np.asarray((X != 0).sum(axis=1)).ravel() / X.shape[1])
     
            X, _ = self._cp10k_log1p_csr(X, scale=self.scale)
            X = X[:, self.keep_]
            Xp = X[:, self.pre_idx_]
     
            mode = self.model_[0]
            if mode == "no_qc":
                return self.model_[1].predict_proba(Xp)
     
            _, svd_pipe, clf = self.model_
            Z = svd_pipe.transform(Xp)
            qc = np.vstack([np.log1p(lib_raw), pct_zeros]).T
            Z2 = np.hstack([Z, qc])
            return clf.predict_proba(Z2)