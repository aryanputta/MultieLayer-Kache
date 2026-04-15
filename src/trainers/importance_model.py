"""
importance_model.py — Trains an XGBoost or LightGBM ranker to predict
KV-cache block importance from the 21-dimensional feature vector.

Training objective: LambdaRank / pairwise ranking (blocks within a request
are ranked by importance; the model learns to order them correctly).

Usage
-----
    trainer = ImportanceModelTrainer(model_type="xgboost")
    metrics = trainer.train(X_train, y_train, X_val, y_val, feature_names)
    trainer.save("results/models/importance_ranker.pkl")
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ImportanceModelTrainer:
    """Trains and evaluates a block-importance ranking model.

    Parameters
    ----------
    model_type:
        ``"xgboost"`` or ``"lightgbm"``.
    config:
        Hyper-parameter overrides.  Recognised keys: ``n_estimators``,
        ``max_depth``, ``learning_rate``, ``subsample``, ``colsample_bytree``,
        ``n_trials`` (Optuna search budget, 0 = skip HPO).
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        config: Optional[dict] = None,
    ) -> None:
        if model_type not in ("xgboost", "lightgbm"):
            raise ValueError(f"model_type must be 'xgboost' or 'lightgbm', got '{model_type}'")
        self.model_type = model_type
        self.config = config or {}
        self._model = None
        self._feature_names: List[str] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[List[str]] = None,
        group_train: Optional[np.ndarray] = None,
        group_val: Optional[np.ndarray] = None,
    ) -> dict:
        """Fit the importance model and return validation metrics.

        Parameters
        ----------
        X_train / X_val:
            Feature matrices of shape (N, n_features).
        y_train / y_val:
            Continuous importance scores in [0, 1].
        feature_names:
            Names for the feature columns (used for importance reporting).
        group_train / group_val:
            Optional request-group arrays for learning-to-rank objectives.
            Each element is the number of blocks belonging to that request.
            If ``None`` the model is trained as a plain regressor.

        Returns
        -------
        dict
            ``val_mse``, ``val_pearson``, ``feature_importance``, ``train_time_s``.
        """
        self._feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]

        n_trials = self.config.get("n_trials", 0)

        if n_trials > 0:
            best_params = self._hpo(X_train, y_train, X_val, y_val, n_trials)
            self.config.update(best_params)

        t0 = time.time()

        if self.model_type == "xgboost":
            self._model = self._build_xgboost(
                X_train, y_train, X_val, y_val,
                group_train, group_val,
            )
        else:
            self._model = self._build_lightgbm(
                X_train, y_train, X_val, y_val,
                group_train, group_val,
            )

        train_time = time.time() - t0
        val_preds = self._model.predict(X_val)
        metrics = self._compute_metrics(y_val, val_preds, feature_names=self._feature_names)
        metrics["train_time_s"] = round(train_time, 2)
        logger.info(
            "ImportanceModelTrainer: trained %s in %.1fs — val_mse=%.5f  val_pearson=%.4f",
            self.model_type,
            train_time,
            metrics["val_mse"],
            metrics["val_pearson"],
        )
        return metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return importance scores in [0, 1] for feature matrix *X*."""
        if self._model is None:
            raise RuntimeError("Model not trained — call train() or load() first.")
        raw = self._model.predict(X)
        lo, hi = float(raw.min()), float(raw.max())
        if hi > lo:
            return ((raw - lo) / (hi - lo)).astype(np.float32)
        return np.full(len(raw), 0.5, dtype=np.float32)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model + feature names to *path* as a pickle file."""
        if self._model is None:
            raise RuntimeError("No model to save.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {"model": self._model, "feature_names": self._feature_names, "model_type": self.model_type}
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info("Model saved to %s", path)

    def load(self, path: str) -> None:
        """Load a previously saved model."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict):
            self._model = payload["model"]
            self._feature_names = payload.get("feature_names", [])
            self.model_type = payload.get("model_type", self.model_type)
        else:
            self._model = payload
        logger.info("Model loaded from %s", path)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> Dict[str, float]:
        """Return a dict of feature_name → importance score."""
        if self._model is None:
            return {}
        try:
            raw = self._model.feature_importances_
        except AttributeError:
            return {}
        names = self._feature_names or [f"f{i}" for i in range(len(raw))]
        total = float(raw.sum()) or 1.0
        return {n: float(v / total) for n, v in zip(names, raw)}

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        k: int = 5,
    ) -> dict:
        """K-fold cross-validation; returns mean/std of MSE and Pearson r."""
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        mse_scores, pearson_scores = [], []
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
            clone = ImportanceModelTrainer(self.model_type, self.config.copy())
            clone.train(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx])
            val_preds = clone.predict(X[va_idx])
            m = self._compute_metrics(y[va_idx], val_preds)
            mse_scores.append(m["val_mse"])
            pearson_scores.append(m["val_pearson"])
        return {
            "k": k,
            "mse_mean": float(np.mean(mse_scores)),
            "mse_std": float(np.std(mse_scores)),
            "pearson_mean": float(np.mean(pearson_scores)),
            "pearson_std": float(np.std(pearson_scores)),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_xgboost(self, X_tr, y_tr, X_va, y_va, g_tr, g_va):
        import xgboost as xgb
        objective = "reg:squarederror"
        if g_tr is not None:
            objective = "rank:pairwise"

        params = {
            "n_estimators": self.config.get("n_estimators", 300),
            "max_depth": self.config.get("max_depth", 6),
            "learning_rate": self.config.get("learning_rate", 0.05),
            "subsample": self.config.get("subsample", 0.8),
            "colsample_bytree": self.config.get("colsample_bytree", 0.8),
            "objective": objective,
            "tree_method": "hist",
            "random_state": 42,
            "verbosity": 0,
        }
        model = xgb.XGBRegressor(**params)

        eval_set = [(X_va, y_va)]
        model.fit(
            X_tr, y_tr,
            eval_set=eval_set,
            verbose=False,
        )
        return model

    def _build_lightgbm(self, X_tr, y_tr, X_va, y_va, g_tr, g_va):
        import lightgbm as lgb
        objective = "regression" if g_tr is None else "lambdarank"
        params = {
            "n_estimators": self.config.get("n_estimators", 300),
            "max_depth": self.config.get("max_depth", 6),
            "learning_rate": self.config.get("learning_rate", 0.05),
            "subsample": self.config.get("subsample", 0.8),
            "colsample_bytree": self.config.get("colsample_bytree", 0.8),
            "objective": objective,
            "random_state": 42,
            "verbose": -1,
        }
        model = lgb.LGBMRegressor(**params)
        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
        eval_set = [(X_va, y_va)]
        model.fit(
            X_tr, y_tr,
            eval_set=eval_set,
            callbacks=callbacks,
        )
        return model

    def _hpo(self, X_tr, y_tr, X_va, y_va, n_trials: int) -> dict:
        """Run Optuna HPO and return best params."""
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            }
            clone = ImportanceModelTrainer(self.model_type, params)
            clone.train(X_tr, y_tr, X_va, y_va)
            preds = clone.predict(X_va)
            return float(np.mean((y_va - preds) ** 2))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        logger.info("HPO best MSE: %.6f", study.best_value)
        return study.best_params

    @staticmethod
    def _compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> dict:
        from scipy.stats import pearsonr
        mse = float(np.mean((y_true - y_pred) ** 2))
        try:
            r, _ = pearsonr(y_true, y_pred)
            r = float(r)
        except Exception:
            r = 0.0
        return {"val_mse": round(mse, 6), "val_pearson": round(r, 4)}
