"""Bernoulli GLM, hierarchical stepwise AIC, glmnet, and the report's evaluation protocol."""
from dataclasses import dataclass
from itertools import chain
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from glmnet import LogitNet
from scipy.special import expit
from scipy.stats import chi2
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

RAW_FEATURES = ["fare", "additional_charges", "pool_index", "part_of_day", "weekend", "route"]
TREE_FEATURES = ["Fare", "Additional.Charges", "Trips.Pooled", "Part.of.Day", "Weekend", "Route.Mean"]
TERMS = {
    "Fare": ["Fare"], "Additional.Charges": ["Additional.Charges"], "Trips.Pooled": ["Trips.Pooled"],
    "Part.of.Day": ["Part.of.Day2", "Part.of.Day3"], "Weekend": ["Weekend1"],
    "Route.Mean": ["Route.Mean"],
    "Part.of.Day:Weekend": ["Part.of.Day2:Weekend1", "Part.of.Day3:Weekend1"],
}
INTERACTION = "Part.of.Day:Weekend"


def balance_and_split(trips, seed=123):
    """R balances before the split. Both holdout partitions therefore have ~50% tips."""
    compact = trips[RAW_FEATURES + ["tip_given"]]
    y = compact.tip_given.to_numpy()
    if len(np.unique(y)) != 2:
        raise ValueError("The analysis requires trips both with and without a recorded tip.")
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == k) for k in (0, 1)]
    size = min(map(len, groups))
    ids = np.concatenate([rng.choice(group, size, replace=False) for group in groups])
    balanced = compact.iloc[ids].reset_index(drop=True)
    train, test = train_test_split(np.arange(len(ids)), train_size=.7,
                                   stratify=balanced.tip_given, random_state=seed)
    return balanced, train, test


@dataclass
class Patterns:
    """One row per predictor/outcome pattern with an exact trip frequency."""
    x: pd.DataFrame
    y: np.ndarray
    w: np.ndarray

    @property
    def n(self):
        return int(self.w.sum())


def encoded_pair(training, testing):
    """Compress repeated values losslessly. Fit route means only on the training rides."""
    def group(frame):
        return frame.groupby(RAW_FEATURES + ["tip_given"], observed=True, sort=False).size().reset_index(name="n")

    train, test = group(training), group(testing)
    hits = (train.tip_given * train.n).groupby(train.route, observed=True).sum()
    counts = train.groupby("route", observed=True).n.sum()
    rates = hits / counts
    fallback = float(hits.sum() / counts.sum())

    def encode(grouped):
        x = grouped[RAW_FEATURES[:-1]].copy()
        x["route"] = grouped.route.map(rates).astype(float).fillna(fallback)
        x.columns = TREE_FEATURES
        return Patterns(x, grouped.tip_given.to_numpy(dtype="int8"), grouped.n.to_numpy(dtype="int64"))

    return encode(train), encode(test), rates.rename("Training tip rate")


def design(patterns):
    x = patterns.x
    part2 = (x["Part.of.Day"].to_numpy() == 2).astype(float)
    part3 = (x["Part.of.Day"].to_numpy() == 3).astype(float)
    weekend = x["Weekend"].to_numpy()
    return pd.DataFrame({
        "Intercept": np.ones(len(x)), "Fare": x["Fare"], "Additional.Charges": x["Additional.Charges"],
        "Trips.Pooled": x["Trips.Pooled"], "Part.of.Day2": part2, "Part.of.Day3": part3,
        "Weekend1": weekend, "Route.Mean": x["Route.Mean"],
        "Part.of.Day2:Weekend1": part2*weekend, "Part.of.Day3:Weekend1": part3*weekend,
    })


def fit_glm(x, y, weights):
    fitted = sm.GLM(y, x, family=sm.families.Binomial(), freq_weights=weights).fit(maxiter=80)
    if not fitted.converged:
        warnings.warn("Binomial GLM did not converge; inspect fit before using its estimates.", stacklevel=2)
    return fitted


def stepwise_aic(x, train, full):
    """Try single-term drops/additions within the full formula; respect marginality."""
    terms = tuple(TERMS)
    active = frozenset(terms)
    current = full
    cache = {active: float(full.aic)}
    history = [{"Change": "Full model", "AIC": full.aic}]

    def columns(selection):
        return ["Intercept", *chain.from_iterable(TERMS[t] for t in terms if t in selection)]

    while True:
        best, best_aic, label = active, current.aic, ""
        for term in terms:
            option = active.symmetric_difference([term])
            if INTERACTION in option and not {"Part.of.Day", "Weekend"} <= option:
                continue
            if option not in cache:
                cache[option] = fit_glm(x[columns(option)], train.y, train.w).aic
            if cache[option] < best_aic - 1e-7:
                best, best_aic = option, cache[option]
                label = ("- " if term in active else "+ ") + term
        if best == active:
            break
        active = best
        current = fit_glm(x[columns(active)], train.y, train.w)
        history.append({"Change": label, "AIC": current.aic})
    df_diff = int(full.df_model - current.df_model)
    statistic = max(0., 2 * (full.llf-current.llf))
    compare = pd.Series({"Reduced AIC": current.aic, "Full AIC": full.aic,
                         "Likelihood-ratio statistic": statistic, "df difference": df_diff,
                         "p-value": chi2.sf(statistic, df_diff) if df_diff > 0 else np.nan})
    return current, pd.DataFrame(history), compare


@dataclass
class LassoFit:
    penalty: float
    coefficient: np.ndarray
    intercept: float

    def predict(self, x):
        return expit(self.intercept + x @ self.coefficient)


def _path(x, y, w, lambdas=None):
    # A weighted Bernoulli row here is the same observation repeated w times.
    keep = w > 0
    solver = LogitNet(alpha=1, n_lambda=100, min_lambda_ratio=.0001, standardize=True,
                      n_splits=0, lambda_path=lambdas, tol=1e-7, max_iter=100_000)
    return solver.fit(x[keep], y[keep], sample_weight=w[keep].astype(float))


def fit_lasso(x, y, w, penalty):
    model = _path(x, y, w, np.array([penalty]))
    return LassoFit(float(penalty), model.coef_path_[0, :, 0], float(model.intercept_path_[0, 0]))


def _fold_weights(weights, k, seed):
    """Shuffle individual ride labels before aggregating them into pattern-by-fold counts."""
    n = int(weights.sum())
    if n < k:
        raise ValueError(f"At least {k} training observations are needed for cross-validation.")
    fold = np.arange(n, dtype=np.intp) % k
    np.random.default_rng(seed).shuffle(fold)
    groups = np.repeat(np.arange(len(weights), dtype=np.intp), weights)
    return np.bincount(groups*k + fold, minlength=len(weights)*k).reshape(-1, k)


def select_lasso(x, train, k=10, seed=123):
    """Select lambda.min by held-out binomial deviance, with a glmnet regularisation path."""
    y, w = train.y, train.w
    master = _path(x, y, w)
    grid = master.lambda_path_
    validation = _fold_weights(w, k, seed)
    deviance = np.zeros((k, len(grid)))
    for j, held in enumerate(validation.T):
        model = _path(x, y, w-held, grid)
        # glmnet can trim the tail of its path. Use its last fitted value beyond that tail.
        # Small blocks keep memory independent of the length of the regularisation path.
        valid = held > 0
        val_x, val_y, freq = x[valid], y[valid], held[valid]
        coefs, offsets = model.coef_path_[0], model.intercept_path_[0]
        for start in range(0, len(grid), 8):
            indices = np.arange(start, min(start+8, len(grid)))
            used = np.minimum(indices, coefs.shape[1]-1)
            logits = val_x @ coefs[:, used] + offsets[used]
            loss = np.logaddexp(0, logits) - val_y[:, None]*logits
            deviance[j, indices] = 2*np.average(loss, axis=0, weights=freq)
    mean = np.average(deviance, axis=0, weights=validation.sum(axis=0))
    best = int(mean.argmin())
    cv = pd.DataFrame({"lambda": grid, "mean_deviance": mean,
                       "standard_error": deviance.std(axis=0, ddof=1)/np.sqrt(k)})
    return fit_lasso(x, y, w, grid[best]), cv


def diagnostics(train, p, lasso=False):
    p = np.clip(np.asarray(p), 1e-15, 1-1e-15)
    prob_observed = np.where(train.y == 1, p, 1-p)
    if lasso:
        # Exact expression in logistic_regression.R: the inactive 0^0 contributes 1.
        magnitude = np.sqrt(2*np.log1p(1/prob_observed))
    else:
        magnitude = np.sqrt(-2*np.log(prob_observed))
    residuals = np.sign(train.y-p) * magnitude
    pearson_mean = np.average((train.y-p) / np.sqrt(p*(1-p)), weights=train.w)
    return residuals, pearson_mean


def classification_results(test, scores):
    matrices, metrics = {}, []
    for name, score in scores.items():
        labels = np.asarray(score) > .5
        y, w = test.y, test.w
        matrices[name] = confusion_matrix(y, labels, labels=[0, 1], sample_weight=w)
        metrics.append({"Model": name, "Accuracy": accuracy_score(y, labels, sample_weight=w),
                        "Precision": precision_score(y, labels, sample_weight=w, zero_division=0),
                        "Recall": recall_score(y, labels, sample_weight=w, zero_division=0),
                        "F1": f1_score(y, labels, sample_weight=w, zero_division=0)})
    return matrices, pd.DataFrame(metrics).set_index("Model")


def cross_validated_rocs(balanced, penalty=None, k=5, seed=123, tree=False):
    """Mean fold AUC and pooled out-of-fold ROC. The R tree ROC uses hard class predictions."""
    from .trees import fit_tree
    # The R code samples fold IDs with replacement, without stratifying the outer folds.
    assignment = np.random.default_rng(seed).integers(k, size=len(balanced))
    collected, records = {}, []
    for fold in range(k):
        train, test, _ = encoded_pair(balanced.iloc[assignment != fold], balanced.iloc[assignment == fold])
        if tree:
            fitted = fit_tree(train, cp=.005)
            predictions = {"Tree": fitted.classify(test.x)}
        else:
            if penalty is None:
                raise ValueError("Provide the training-selected glmnet lambda for the logistic ROC.")
            x, val = design(train), design(test)
            glm = fit_glm(x, train.y, train.w)
            lasso = fit_lasso(x.iloc[:, 1:].to_numpy(), train.y, train.w, penalty)
            predictions = {"Logistic Regression": np.asarray(glm.predict(val)),
                           "Lasso Regression": lasso.predict(val.iloc[:, 1:].to_numpy())}
        for name, score in predictions.items():
            auc = roc_auc_score(test.y, score, sample_weight=test.w)
            records.append({"Fold": fold+1, "Model": name, "AUC": auc})
            collected.setdefault(name, []).append((test.y, score, test.w))
    curves = {}
    for name, parts in collected.items():
        truth, scores, weights = (np.concatenate(block) for block in zip(*parts))
        fpr, tpr, _ = roc_curve(truth, scores, sample_weight=weights)
        curves[name] = fpr, tpr
    return curves, pd.DataFrame(records)
