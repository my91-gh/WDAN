from sklearn.metrics import r2_score
import numpy as np


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def SMAPE(pred, true):
    return np.mean(2 * np.abs((pred - true)) / (np.abs(pred) + np.abs(true)))


def metrics_np(pred, true, mask_value='none'):
    if mask_value != 'none':
        # print('metrics_np with mask_value:', mask_value)
        mask = np.abs(true) > (mask_value + 1e-20)
        pred = pred[mask]
        true = true[mask]
    mae = MAE(pred, true).astype(np.float64)
    mse = MSE(pred, true).astype(np.float64)
    rmse = RMSE(pred, true).astype(np.float64)
    smape = SMAPE(pred, true).astype(np.float64)
    r2 = r2_score(true.reshape(-1), pred.reshape(-1))
    return mae, mse, rmse, smape, r2