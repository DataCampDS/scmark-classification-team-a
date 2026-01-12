import numpy as np
from scipy import sparse
     
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
     
     
class Classifier:
        def __init__(
            self,
            pre_k=3000,        # pré-filtrage dispersion
            C_l1=0.3,          # force L1
            C_l2=3.0,          # force L2 final
            n_components=50,   # PCA
            random_state=42,
        ):
            self.pre_k = pre_k
            self.C_l1 = C_l1
            self.C_l2 = C_l2
            self.n_components = n_components
            self.random_state = random_state
     
            self.pre_idx_ = None
            self.l1_mask_ = None
            self.model_ = None
            self.classes_ = None
     
     
        @staticmethod
        def _log1p(X):
            if sparse.issparse(X):
                X = X.tocsr().copy()
                X.data = np.log1p(X.data)
                return X
            return np.log1p(X)
     
        @staticmethod
        def _dispersion(X):
     
            if sparse.issparse(X):
                mean = np.asarray(X.mean(axis=0)).ravel()
                mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
                var = mean_sq - mean**2
            else:
                mean = X.mean(axis=0)
                var = X.var(axis=0)
     
            var[var < 0] = 0.0
            return var / (mean + 1e-6)
     
       
        def fit(self, X, y):
            # log1p 
            X = self._log1p(X)
     
            # pré-filtrage par dispersion 
            disp = self._dispersion(X)
            k = min(self.pre_k, disp.shape[0])
     
            idx = np.argpartition(disp, -k)[-k:]
            idx = idx[np.argsort(disp[idx])[::-1]]
            self.pre_idx_ = idx
     
            X_pre = X[:, idx]
            if sparse.issparse(X_pre):
                X_pre = X_pre.toarray()
     
            #L1 : sélection supervisée 
            l1 = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    penalty="l1",
                    solver="saga",
                    C=self.C_l1,
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=self.random_state
                ))
            ])
     
            l1.fit(X_pre, y)
            coef = l1.named_steps["clf"].coef_
            mask = np.any(coef != 0, axis=0)
     
            
            if mask.sum() < 20:
                mask[:] = True
     
            self.l1_mask_ = mask
            X_sel = X_pre[:, mask]
     
            # modèle final L2 
            n_comp = min(self.n_components, max(1, X_sel.shape[1] - 1))
     
            self.model_ = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(
                    n_components=n_comp,
                    random_state=self.random_state
                )),
                ("clf", LogisticRegression(
                    penalty="l2",
                    C=self.C_l2,
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=self.random_state
                ))
            ])
     
            self.model_.fit(X_sel, y)
            self.classes_ = self.model_.classes_
            return self
     
        def predict_proba(self, X):
            X = self._log1p(X)
     
            X_pre = X[:, self.pre_idx_]
            if sparse.issparse(X_pre):
                X_pre = X_pre.toarray()
     
            X_sel = X_pre[:, self.l1_mask_]
            return self.model_.predict_proba(X_sel)