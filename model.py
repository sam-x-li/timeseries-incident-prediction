
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

RNG_SEED = 1 


class lightGBM():
    def __init__(self):
        self.model = LGBMClassifier(
            objective="binary",
            n_estimators=400,       # max number of trees
            learning_rate=0.05,
            num_leaves=31,
            is_unbalance=True,      
            subsample=0.8,          # like dropout
            colsample_bytree=0.8,   # like dropout
            random_state=RNG_SEED,         # rng seed
            verbose=-1,             
        )
    
    def train(
            self,
            x_train: pd.DataFrame,
            y_train: pd.Series,
            x_eval: pd.DataFrame,
            y_eval: pd.Series,
    ):
        self.model.fit(
            x_train, y_train,
            eval_set=[(x_eval, y_eval)],
            callbacks=[early_stopping(40, verbose=False)]
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

class logisticRegressionWrapper():
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RNG_SEED)

    def train(self,
              x_train: pd.DataFrame,
              y_train: pd.Series,
    ):
        X_scaled = self.scaler.fit_transform(x_train)
        self.model.fit(X_scaled, y_train)
    
    def predict(self, X: pd.DataFrame):
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]


# ── Threshold selection ───────────────────────────────────────────────────────

def select_threshold(y_true: pd.Series, proba: np.ndarray) -> float:
    """
    Sweep decision thresholds and return the one with the best F1.
    Evaluated on the validation set — never on the test set.
    """
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.linspace(0.05, 0.95, 181):
        preds = (proba >= thr).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1, best_thr = score, float(thr)
    return best_thr


# ── Persistence ───────────────────────────────────────────────────────────────

def save(obj: object, path: str) -> None:
    joblib.dump(obj, path)


def load(path: str) -> object:
    return joblib.load(path)
