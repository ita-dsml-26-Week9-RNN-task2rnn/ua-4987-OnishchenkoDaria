from __future__ import annotations

"""Task 2 — Multi-step forecasting strategies (Keras).

Goal
----
Compare forecast drift on horizon H=100 using:
1) One-step model (predicts x[t+1]) rolled out recursively with stride=1.
2) K-step model (K=20, predicts x[t+1:t+K]) rolled out recursively with stride=20.
3) The same K-step model rolled out with stride=1, using only the first predicted step each time.

Students implement the core pipeline:
- make_windows
- time_split
- build_model
- train_model
- recursive_rollout_one_step
- recursive_rollout_k_step_stride_k
- recursive_rollout_k_step_stride_1

Everything else (metrics, evaluation helpers, demo plotting) is provided.

Important
---------
- Use time-based split (NO shuffle) to avoid data leakage.
- The difference between strategies is *inference-time usage*, not only the model.
"""

from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# ----------------------------
# Metrics (provided)
# ----------------------------

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth values. Shape: (N,) or (N, 1).
    y_pred : np.ndarray
        Predicted values. Same shape as y_true.

    Returns
    -------
    float
        MAE.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth values. Shape: (N,) or (N, 1).
    y_pred : np.ndarray
        Predicted values. Same shape as y_true.

    Returns
    -------
    float
        RMSE.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ----------------------------
# Data helpers (students implement)
# ----------------------------

def make_windows(series: np.ndarray, window: int, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Create supervised windows for forecasting.

    For each start index t, we build:
        X[t] = series[t : t+window]
        y[t] = series[t+window : t+window+horizon]

    This supports both:
    - one-step forecasting: horizon=1, y shape (N, 1)
    - K-step forecasting: horizon=K, y shape (N, K)

    Parameters
    ----------
    series : np.ndarray
        1D time series of length T. Shape: (T,).
    window : int
        Window length w. Must satisfy 1 <= window < T.
    horizon : int
        Forecast horizon K (number of future steps per sample). Must satisfy horizon >= 1.

    Returns
    -------
    X : np.ndarray
        Input windows with feature dimension for Keras RNN layers.
        Shape: (N, window, 1).
    y : np.ndarray
        Targets.
        - If horizon=1: shape (N, 1)
        - If horizon>1: shape (N, horizon)

    Notes
    -----
    The number of samples is N = T - window - horizon + 1.
    """
    series = np.asarray(series, dtype=np.float32)
    if window < 1 or window >= len(series):
        raise ValueError("window must satisfy 1 <= window < len(series)")
    series = np.asarray(series, dtype=np.float32)
    if horizon < 1:
        raise ValueError("horizon must satisfy horizon >= 1")

    n_samples = len(series) - window - horizon + 1
    if n_samples < 1:
        raise ValueError(
            "series too short for given window and horizon "
            f"(need at least {window + horizon} points, got {len(series)})"
        )

    X, y = [], []
    for i in range(n_samples):
        X.append(series[i: i + window])
        y.append(series[i + window: i + window + horizon])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X[..., None], y


def time_split(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Time-based split (NO shuffle) to avoid leakage.

    Parameters
    ----------
    X : np.ndarray
        Windowed inputs, shape (N, window, 1).
    y : np.ndarray
        Targets, shape (N, 1) for horizon=1 or (N, K) for horizon=K.
    train_frac : float
        Train fraction.
    val_frac : float
        Validation fraction.

    Returns
    -------
    (X_train, y_train), (X_val, y_val), (X_test, y_test)

    Raises
    ------
    ValueError
        If fractions are invalid or produce empty splits.
    """
    if not (0 < train_frac < 1) or not (0 <= val_frac < 1):
        raise ValueError("train_frac must be in (0,1) and val_frac in [0,1)")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1 (leave room for test)")

    n = len(X)
    train_end = int(train_frac * n)
    val_end = int((train_frac + val_frac) * n)

    if train_end == 0 or val_end == train_end or val_end == n:
        raise ValueError("One of the splits is empty; adjust fractions or provide more data")

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ----------------------------
# Model helpers (students implement)
# ----------------------------

def build_model(
    window: int,
    output_dim: int,
    n_units: int = 64,
    dense_units: int = 32,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Build and compile an LSTM model.

    Parameters
    ----------
    window : int
        Input window length. Input shape will be (window, 1).
    output_dim : int
        Number of outputs.
        - output_dim=1 for one-step model
        - output_dim=K for K-step model
    n_units : int
        LSTM units.
    dense_units : int
        Dense hidden units.
    dropout : float
        Dropout after LSTM.
    learning_rate : float
        Adam learning rate.

    Returns
    -------
    tf.keras.Model
        Compiled model with output shape (None, output_dim).

    Notes
    -----
    - For output_dim>1, use a Dense(output_dim) output layer (vector prediction).
    - Keep loss as MSE, metric MAE.
    """
    model = Sequential([
        LSTM(n_units, input_shape=(window, 1)),
        Dropout(dropout),
        Dense(dense_units, activation="relu"),
        Dense(output_dim),
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def train_model(
    series: np.ndarray,
    window: int,
    horizon: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    epochs: int = 25,
    batch_size: int = 64,
    seed: int = 42,
    verbose: int = 0,
) -> Tuple[tf.keras.Model, np.ndarray, np.ndarray]:
    """Train a model for the given horizon and return model + test split.

    Parameters
    ----------
    series : np.ndarray
        1D time series, shape (T,).
    window : int
        Window length.
    horizon : int
        Forecast horizon per sample.
        - 1 for one-step model
        - K (e.g., 20) for K-step model
    train_frac : float
        Train fraction.
    val_frac : float
        Validation fraction.
    epochs : int
        Training epochs.
    batch_size : int
        Batch size.
    seed : int
        Random seed.
    verbose : int
        Verbosity for training.

    Returns
    -------
    model : tf.keras.Model
        Trained model.
    X_test : np.ndarray
        Test windows, shape (N_test, window, 1).
    y_test : np.ndarray
        Test targets, shape (N_test, 1) or (N_test, K).

    Notes
    -----
    - Use EarlyStopping (recommended) to reduce overfitting.
    - Do not shuffle time.
    """
    tf.keras.utils.set_random_seed(seed)

    X, y = make_windows(series, window=window, horizon=horizon)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = time_split(
        X, y, train_frac=train_frac, val_frac=val_frac
    )

    model = build_model(window=window, output_dim=horizon)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        )
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=callbacks,
    )

    return model, X_test, y_test


# ----------------------------
# Rollout strategies (students implement)
# ----------------------------

def recursive_rollout_one_step(
    model: tf.keras.Model,
    init_window: np.ndarray,
    horizon: int = 100,
) -> np.ndarray:
    """Recursive rollout for a one-step model (stride=1).

    Parameters
    ----------
    model : tf.keras.Model
        One-step model that maps (1, window, 1) -> (1, 1).
    init_window : np.ndarray
        Initial context window (seed). Shape: (window,).
    horizon : int
        Number of future steps to generate.

    Returns
    -------
    np.ndarray
        Forecast of length `horizon`. Shape: (horizon,).

    Notes
    -----
    At each step we:
    1) predict next value
    2) append it to the window
    3) shift window by 1
    """
    window = np.asarray(init_window, dtype=np.float32).copy()
    preds = []

    for _ in range(horizon):
        x_in = window[np.newaxis, :, np.newaxis]  # (1, window, 1)
        next_val = model.predict(x_in, verbose=0)[0, 0]
        preds.append(next_val)
        window = np.concatenate([window[1:], [next_val]])

    return np.array(preds, dtype=np.float32)


def recursive_rollout_k_step_stride_k(
    model: tf.keras.Model,
    init_window: np.ndarray,
    k: int = 20,
    horizon: int = 100,
) -> np.ndarray:
    """Recursive rollout for a K-step model using stride=K.

    Parameters
    ----------
    model : tf.keras.Model
        K-step model that maps (1, window, 1) -> (1, K).
    init_window : np.ndarray
        Initial context window. Shape: (window,).
    k : int
        Steps predicted per model call.
    horizon : int
        Total steps to generate. Assumed divisible by k (e.g., 100 with k=20).

    Returns
    -------
    np.ndarray
        Forecast of length `horizon`. Shape: (horizon,).

    Notes
    -----
    We repeatedly:
    1) predict a block of K future values
    2) append the full block
    3) shift window by K

    This reduces recursion depth (H/K calls).
    """
    if horizon % k != 0:
        raise ValueError("horizon must be divisible by k for stride=k rollout")

    window = np.asarray(init_window, dtype=np.float32).copy()
    preds = []
    n_calls = horizon // k

    for _ in range(n_calls):
        x_in = window[np.newaxis, :, np.newaxis]
        block = model.predict(x_in, verbose=0)[0]
        preds.append(block)
        window = np.concatenate([window[k:], block])

    return np.concatenate(preds).astype(np.float32)


def recursive_rollout_k_step_stride_1(
    model: tf.keras.Model,
    init_window: np.ndarray,
    k: int = 20,
    horizon: int = 100,
) -> np.ndarray:
    """Recursive rollout for a K-step model using stride=1.

    Parameters
    ----------
    model : tf.keras.Model
        K-step model that maps (1, window, 1) -> (1, K).
    init_window : np.ndarray
        Initial context window. Shape: (window,).
    k : int
        Steps predicted per model call.
    horizon : int
        Total steps to generate.

    Returns
    -------
    np.ndarray
        Forecast of length `horizon`. Shape: (horizon,).

    Notes
    -----
    At each step we:
    1) predict K future values
    2) take ONLY the first predicted value (t+1)
    3) append it
    4) shift window by 1

    This uses the K-step model as a one-step generator.
    """
    window = np.asarray(init_window, dtype=np.float32).copy()
    preds = []

    for _ in range(horizon):
        x_in = window[np.newaxis, :, np.newaxis]
        block = model.predict(x_in, verbose=0)[0]
        next_val = block[0]
        preds.append(next_val)
        window = np.concatenate([window[1:], [next_val]])

    return np.array(preds, dtype=np.float32)


# ----------------------------
# Evaluation + plotting (provided)
# ----------------------------

def horizon_errors(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE/RMSE for a horizon forecast.

    Parameters
    ----------
    y_true : np.ndarray
        True future values, shape (H,).
    y_pred : np.ndarray
        Predicted future values, shape (H,).

    Returns
    -------
    dict
        {"mae": float, "rmse": float}
    """
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred)}


def plot_rollouts(y_true: np.ndarray, preds: Dict[str, np.ndarray]) -> None:
    """Plot ground truth and multiple forecast rollouts.

    Parameters
    ----------
    y_true : np.ndarray
        True future values, shape (H,).
    preds : Dict[str, np.ndarray]
        Mapping strategy_name -> predicted future values, each shape (H,).
    """
    plt.figure(figsize=(12, 4))
    plt.plot(y_true, label="true", linewidth=2)
    for name, y_hat in preds.items():
        plt.plot(y_hat, label=name, alpha=0.9)
    plt.grid(True)
    plt.legend()
    plt.title("Multi-step rollout comparison")
    plt.show()


# ----------------------------
# Demo (provided, not used in tests)
# ----------------------------

def _make_series(n: int = 2500, seed: int = 0) -> np.ndarray:
    """Generate a synthetic series (trend + seasonality + noise)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float32)
    x = 0.0009 * t + 2.0 * np.sin(2 * np.pi * t / 50.0) + 0.8 * np.sin(2 * np.pi * t / 16.0)
    x += rng.normal(0, 0.2, size=n).astype(np.float32)
    return x.astype(np.float32)


def demo() -> None:
    """End-to-end demo.

    Trains a one-step model and a K-step model (K=20) on synthetic data,
    then compares three rollout strategies on horizon H=100.

    This function is for student orientation (plots), not for unit tests.
    """
    tf.keras.utils.set_random_seed(123)

    series = _make_series(n=2600, seed=123)
    window = 40
    k = 20
    H = 100

    # Train models (students implement train_model)
    one_model, X_test_1, y_test_1 = train_model(series, window=window, horizon=1, epochs=15, seed=123, verbose=0)
    k_model, X_test_k, y_test_k = train_model(series, window=window, horizon=k, epochs=15, seed=123, verbose=0)

    # Create an initial window and ground-truth future from the end of the series
    init_window = series[-(window + H) : -H]
    y_true = series[-H:]

    pred_1 = recursive_rollout_one_step(one_model, init_window, horizon=H)
    pred_k20 = recursive_rollout_k_step_stride_k(k_model, init_window, k=k, horizon=H)
    pred_k1 = recursive_rollout_k_step_stride_1(k_model, init_window, k=k, horizon=H)

    preds = {
        "one-step (stride=1)": pred_1,
        "K-step=20 (stride=20)": pred_k20,
        "K-step=20 (stride=1, use first)": pred_k1,
    }

    for name, y_hat in preds.items():
        print(name, horizon_errors(y_true, y_hat))

    plot_rollouts(y_true, preds)


if __name__ == "__main__":
    demo()
