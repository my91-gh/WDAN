#%
import json
import os

import numpy as np
import optuna
import pandas as pd
import ta
import vectorbt as vbt
from lightgbm import LGBMClassifier  # type: ignore
from sklearn.ensemble import AdaBoostClassifier  # noqa: F401
from sklearn.ensemble import ExtraTreesClassifier  # noqa: F401
from sklearn.ensemble import GradientBoostingClassifier  # noqa: F401
from sklearn.ensemble import RandomForestClassifier  # noqa: F401
from sklearn.naive_bayes import GaussianNB  # noqa: F401
from sklearn.tree import DecisionTreeClassifier  # noqa: F401
from xgboost import XGBClassifier  # type: ignore

# %%
# 1. Prepare the data
SYMBOL = "BTCUSDT"
START_DATE = "2020-01-01"
END_DATE = "2025-06-15"

# Attempt to cache raw price data (cheap, but keeps things tidy)
data = vbt.BinanceData.download(
    SYMBOL, start=START_DATE, end=END_DATE, interval="1d"
).get(["Open", "High", "Low", "Close", "Volume"])

# ------------------------------------------------------------
# Feature caching
# ------------------------------------------------------------
FEATURE_FILE = f"features_{SYMBOL}_{START_DATE}_{END_DATE}.parquet"

if os.path.exists(FEATURE_FILE):
    print(f"Loading cached features from {FEATURE_FILE}")
    feature_df = pd.read_parquet(FEATURE_FILE)
else:
    print("Generating features – first run or cache miss.")

    def calculate_features(df):
        """Add full TA feature set and basic derived columns."""
        df = df.copy()

        df = ta.add_all_ta_features(
            df,
            open="Open",
            high="High",
            low="Low",
            close="Close",
            volume="Volume",
            fillna=True,
        )

        df["returns"] = df["Close"].pct_change()
        df["target"] = (df["returns"].shift(-1) > 0).astype(int)

        return df.dropna()

    feature_df = calculate_features(pd.DataFrame(data))
    feature_df.to_parquet(FEATURE_FILE)
    print(f"Saved features to {FEATURE_FILE}")

# %%
# 3. Hyperparameter optimization with Optuna

# Metrics available from vectorbt stats that make sense as objectives
TARGET_STAT = "Calmar Ratio"  # can be changed to Sharpe Ratio, Total Return [%], etc.

# Precompute correlation ranking once
_CORR_RANKING = (
    feature_df.corr()["target"].drop("target").abs().sort_values(ascending=False)
)

# FEATURE_COLUMNS will be decided per Optuna trial based on hyperparameter top_n

# Mapping of classifier names to classes (optional libs guarded)
CLASSIFIERS = {
    "AdaBoost": AdaBoostClassifier,
    "LightGBM": LGBMClassifier,
    "XGBoost": XGBClassifier,
}


def objective(trial):
    # Hyperparameter search space (narrowed)
    train_window = trial.suggest_int("train_window", 150, 400, step=50)
    test_window = trial.suggest_int("test_window", 30, 150, step=30)

    # top N features hyperparameter (keep static 10 to focus on model search)
    top_n = 10
    feature_columns = _CORR_RANKING.head(top_n).index.tolist()

    # choose classifier
    classifier_name = trial.suggest_categorical("classifier", list(CLASSIFIERS.keys()))

    model_params = {}

    if classifier_name == "AdaBoost":
        model_params = {
            "n_estimators": trial.suggest_int("ada_n_estimators", 250, 400),
            "learning_rate": trial.suggest_float("ada_lr", 0.3, 1.0),
        }
    elif classifier_name == "LightGBM":
        model_params = {
            "n_estimators": trial.suggest_int("lgb_n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("lgb_lr", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("lgb_leaves", 15, 63),
            "max_depth": trial.suggest_int("lgb_max_depth", 3, 10),
        }
    elif classifier_name == "XGBoost":
        model_params = {
            "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("xgb_lr", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("xgb_max_depth", 3, 10),
            "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("xgb_colsample", 0.6, 1.0),
            "gamma": trial.suggest_float("xgb_gamma", 0, 5),
        }

    # Fee narrowed to values that performed best (mostly 0.0005)
    fee = trial.suggest_categorical("fee", [0.0005])

    predictions = pd.Series(index=feature_df.index, dtype=float)

    # Rolling-window training / prediction
    for start in range(train_window, len(feature_df), test_window):
        train_start = start - train_window
        test_end = min(start + test_window, len(feature_df))

        X_train = feature_df.iloc[train_start:start][feature_columns]
        y_train = feature_df.iloc[train_start:start]["target"]
        X_test = feature_df.iloc[start:test_end][feature_columns]

        model_cls = CLASSIFIERS[classifier_name]
        model = model_cls(**model_params)
        model.fit(X_train, y_train)
        predictions.iloc[start:test_end] = model.predict(X_test)

    # If not enough predictions were made (e.g., due to window sizes), penalize trial
    if predictions.notna().sum() == 0:
        return 0.0

    # Attach predictions to dataframe copy to avoid mutating global
    df = feature_df.copy()
    df["prediction"] = predictions

    # Create trading signals
    df["signal"] = np.where(
        (df["prediction"] == 1),
        1,
        np.where(
            (df["prediction"] == 0),
            -1,
            0,
        ),
    )

    close = df["Close"]
    signal = df["signal"]

    if signal.abs().sum() == 0:
        return 0.0  # no trades – bad hyperparams

    pf = vbt.Portfolio.from_signals(
        close,
        entries=signal == 1,
        exits=signal == -1,
        fees=fee,
        freq="1D",
    )

    stats = pf.stats()

    # Some stats may come back as a DataFrame if multiple portfolios; get scalar
    score = float(stats.loc[TARGET_STAT]) if TARGET_STAT in stats.index else 0.0

    # Report intermediate value (Optuna dashboard etc.)
    trial.report(score, step=0)

    return score


PARAM_FILE = "best_params_ada.json"

if os.path.exists(PARAM_FILE):
    with open(PARAM_FILE, "r") as fp:
        best_params = json.load(fp)
    print(f"Loaded best parameters from {PARAM_FILE}, skipping optimization.")
    study = None

    # ensure rolling-window params exist (older cache files may lack them)
    if "train_window" not in best_params or "test_window" not in best_params:
        print(
            "[WARN] Cached params missing rolling-window keys – setting defaults (train_window=200, test_window=100)."
        )
        best_params.setdefault("train_window", 200)
        best_params.setdefault("test_window", 100)
        # re-save updated cache for future runs
        with open(PARAM_FILE, "w") as fp:
            json.dump(best_params, fp)
else:
    N_TRIALS = 100
    n_jobs = max(os.cpu_count() - 1, 1)
    study = optuna.create_study(direction="maximize", study_name="rf_vectorbt")
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=n_jobs, show_progress_bar=True)

    print("Best trial:")
    print("  Value ({}):".format(TARGET_STAT), study.best_value)
    print("  Params:")
    for k, v in study.best_params.items():
        print("    {}: {}".format(k, v))

    # Save best params
    with open(PARAM_FILE, "w") as fp:
        json.dump(study.best_params, fp)
    best_params = study.best_params

    # Print all trials sorted by value descending
    print("\nAll trials (sorted):")
    sorted_trials = sorted(
        study.trials,
        key=lambda t: t.value if t.value is not None else float("-inf"),
        reverse=True,
    )
    for t in sorted_trials:
        print(f"Trial {t.number}: value={t.value:.6f} params={t.params}")

# %%
# 4. Run backtest with best params and display detailed stats/plots
print("\nRunning backtest with best parameters...")
print(best_params)

# Re-run objective logic quickly to capture best pf object for analysis
# -----
predictions = pd.Series(index=feature_df.index, dtype=float)
for start in range(
    best_params["train_window"], len(feature_df), best_params["test_window"]
):
    train_start = start - best_params["train_window"]
    test_end = min(start + best_params["test_window"], len(feature_df))

    # Determine feature columns from best top_n
    best_top_n = best_params.get("top_n", 10)
    feature_cols_best = _CORR_RANKING.head(best_top_n).index.tolist()

    X_train = feature_df.iloc[train_start:start][feature_cols_best]
    y_train = feature_df.iloc[train_start:start]["target"]
    X_test = feature_df.iloc[start:test_end][feature_cols_best]

    # Build model for final evaluation from best_params
    clf_name = best_params.get("classifier", "AdaBoost")

    # Helper to extract params with prefix
    def params_with_prefix(prefix):
        return {
            k.replace(prefix, ""): v
            for k, v in best_params.items()
            if k.startswith(prefix)
        }

    if clf_name == "RandomForest":
        model_kwargs = params_with_prefix("rf_")
        model_kwargs["random_state"] = 42
        model_kwargs["class_weight"] = "balanced"
    elif clf_name == "GradientBoosting":
        model_kwargs = params_with_prefix("gb_")
        # Rename shorthand key if present
        if "lr" in model_kwargs:
            model_kwargs["learning_rate"] = model_kwargs.pop("lr")
    elif clf_name == "ExtraTrees":
        model_kwargs = params_with_prefix("et_")
        model_kwargs["random_state"] = 42
    elif clf_name == "AdaBoost":
        model_kwargs = params_with_prefix("ada_")
        if "lr" in model_kwargs:
            model_kwargs["learning_rate"] = model_kwargs.pop("lr")
    elif clf_name == "LogisticRegression":
        model_kwargs = params_with_prefix("lr_")
        model_kwargs["max_iter"] = 1000
    elif clf_name == "SVC":
        model_kwargs = params_with_prefix("svc_")
        model_kwargs["gamma"] = "scale"
        model_kwargs["probability"] = True
    elif clf_name == "KNN":
        model_kwargs = params_with_prefix("knn_")
    elif clf_name == "DecisionTree":
        model_kwargs = params_with_prefix("dt_")
        model_kwargs["random_state"] = 42
    elif clf_name == "LightGBM":
        model_kwargs = params_with_prefix("lgb_")
    elif clf_name == "XGBoost":
        model_kwargs = params_with_prefix("xgb_")
    else:
        model_kwargs = {}

    model = CLASSIFIERS[clf_name](**model_kwargs)
    model.fit(X_train, y_train)
    predictions.iloc[start:test_end] = model.predict(X_test)

df = feature_df.copy()
df["prediction"] = predictions

# ---------------------------------------------------------------------------
# Regime filter: flat when 200-EMA slope is negative **or** 30-day vol is high
# ---------------------------------------------------------------------------
# 200-day EMA and its slope
df["ema200"] = df["Close"].ewm(span=200).mean()
df["ema_slope"] = df["ema200"].diff()

# 30-day realized volatility (std of log returns)
df["realised_vol"] = np.log(df["Close"]).diff().rolling(30).std()

# Good regime condition
VOL_THRESHOLD = 0.06  # ~6 % daily vol (adjust as needed)
df["good_regime"] = (df["ema_slope"] > 0) & (df["realised_vol"] < VOL_THRESHOLD)

# ---------------------------------------------------------------------------
# Raw directional signal from classifier
# ---------------------------------------------------------------------------
df["raw_signal"] = np.where(
    (df["prediction"] == 1),
    1,
    np.where(df["prediction"] == 0, -1, 0),
)

# Apply regime filter – flat (0) outside good regimes
df["signal"] = df["raw_signal"] * df["good_regime"].astype(int)

# ---------------------------------------------------------------------------
# ATR-based position sizing (risk = 2 % of equity per trade)
# ---------------------------------------------------------------------------
from ta.volatility import AverageTrueRange  # noqa: E402

atr = AverageTrueRange(
    high=df["High"], low=df["Low"], close=df["Close"], window=14, fillna=True
).average_true_range()

RISK_PCT = 0.02  # risk 2 % of equity per trade
INIT_EQUITY = 10_000  # starting equity in quote currency

# Position size (number of BTC) per entry day; NaN/0 elsewhere
position_size = (RISK_PCT * INIT_EQUITY) / atr

# -------------------------------
# Split signals into long/short masks (vectorbt requires direction via masks)
# -------------------------------
long_entries = df["signal"] == 1
long_exits = (df["signal"].shift(1) == 1) & (df["signal"] != 1)

short_entries = df["signal"] == -1
short_exits = (df["signal"].shift(1) == -1) & (df["signal"] != -1)

# Use *positive* sizing – same magnitude for long and short positions
size = position_size.abs()

pf = vbt.Portfolio.from_signals(
    df["Close"],
    entries=long_entries,
    exits=long_exits,
    short_entries=short_entries,
    short_exits=short_exits,
    size=size,
    fees=best_params["fee"],
    init_cash=INIT_EQUITY,
    freq="1D",
)

print("\nBacktest stats with regime filter & ATR sizing:")
print(pf.stats())

pf.plot().show()

# %%