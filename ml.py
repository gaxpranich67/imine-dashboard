"""Machine-learning predictors for imine conversion.

Reactants are described by physical-organic descriptors rather than by name, so the models can be
queried for substituents and amines that are not in the training set:
    sigma_p, sigma_p_plus  (aldehyde para-substituent, Hammett)
    taft_Es                (N-alkyl group of the amine, Taft steric constant)
    mtbe                   (solvent indicator: 0 = DCM, 1 = MTBE)
    inv_T                  (1000 / T, K^-1)

Two models are offered:
  * Kinetic GP: a Gaussian process learns ln k_app from the descriptors; conversion then follows from the
    fitted rate law dx/dt = k_app (1 - x)^n. Smooth in time, monotonic, with a predictive uncertainty.
  * Direct GBM: gradient-boosted trees predict conversion (%) from the descriptors plus time, constrained to
    increase with time and temperature.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis import conversion, fit_kinetics

DESCRIPTORS = ['sigma_p', 'sigma_p_plus', 'taft_Es', 'mtbe', 'inv_T']


def add_descriptors(frame, sigma_p, sigma_p_plus, taft_es):
    """Attach descriptor columns to rows that carry aldehyde_R_group, amine_name, solvent and temperature_C."""
    out = frame.copy()
    if 'sigma_p' not in out:
        out['sigma_p'] = out.aldehyde_R_group.map(sigma_p)
    if 'sigma_p_plus' not in out:
        out['sigma_p_plus'] = out.aldehyde_R_group.map(sigma_p_plus) if 'aldehyde_R_group' in out else out.sigma_p
    if 'taft_Es' not in out:
        out['taft_Es'] = out.amine_name.map(taft_es)
    out['mtbe'] = (out.solvent.astype(str).str.upper() == 'MTBE').astype(float)
    out['inv_T'] = 1000 / (out.temperature_C.astype(float) + 273.15)
    return out


def _gp():
    kernel = (ConstantKernel(1.0, (1e-2, 1e3)) * RBF(length_scale=np.ones(len(DESCRIPTORS)), length_scale_bounds=(1e-2, 1e3))
              + WhiteKernel(1e-3, (1e-6, 1e0)))
    return make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=3, random_state=0))


def _gbm():
    # Monotonic constraints: conversion rises with time and with temperature (falls with 1000/T).
    cst = [0, 0, 0, 0, -1, 1]
    return HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=4,
                                         l2_regularization=1e-3, monotonic_cst=cst, random_state=0)


def train_kinetic_gp(kin):
    """kin: fitted rate constants with descriptor columns. Target is ln k_app (min^-1)."""
    model = _gp()
    model.fit(kin[DESCRIPTORS].to_numpy(), np.log(kin.k_app_per_min.to_numpy()))
    return model


def train_direct_gbm(rows):
    model = _gbm()
    model.fit(rows[DESCRIPTORS + ['time_min']].to_numpy(), rows.conversion_percent.to_numpy())
    return model


def predict_ln_k(model, features):
    mean, std = model.predict(np.asarray(features, float), return_std=True)
    return mean, std


def predict_conversion_gp(model, features, times, order):
    """Mean and 95% band (percent) for each feature row at each time (rows x times)."""
    mean, std = predict_ln_k(model, features)
    t = np.asarray(times, float)[None, :]
    centre = conversion(t, np.exp(mean)[:, None], order) * 100
    low = conversion(t, np.exp(mean - 1.96 * std)[:, None], order) * 100
    high = conversion(t, np.exp(mean + 1.96 * std)[:, None], order) * 100
    return centre, low, high, mean, std


def predict_conversion_gbm(model, features, times):
    features = np.asarray(features, float)
    times = np.asarray(times, float)
    grid = np.column_stack([np.repeat(features, len(times), axis=0), np.tile(times, len(features))])
    return np.clip(model.predict(grid), 0, 100).reshape(len(features), len(times))


def cross_validate(data, order, sigma_p, sigma_p_plus, taft_es):
    """Out-of-sample conversion predictions for both models under two splitting schemes.

    Random 5-fold     - new conditions for reactants the model has seen (interpolation).
    Leave-1-aldehyde  - an aldehyde the model has never seen (new substrate).
    Kinetic GP folds are made over whole time courses, so no course is split between train and test.
    """
    rows = add_descriptors(data, sigma_p, sigma_p_plus, taft_es).dropna(subset=DESCRIPTORS).reset_index(drop=True)
    kin = add_descriptors(fit_kinetics(rows, order), sigma_p, sigma_p_plus, taft_es).reset_index(drop=True)
    key = ['aldehyde_name', 'amine_name', 'solvent', 'temperature_C']
    course_id = rows.set_index(key).index.map({k: i for i, k in enumerate(kin.set_index(key).index)}).to_numpy()
    schemes = {'Random 5-fold (new conditions)': list(KFold(5, shuffle=True, random_state=0).split(kin)),
               'Leave one aldehyde out (new substrate)': list(GroupKFold(kin.aldehyde_name.nunique()).split(kin, groups=kin.aldehyde_name))}
    records, preds = [], []
    for scheme, folds in schemes.items():
        gp_pred = np.full(len(rows), np.nan)
        gbm_pred = np.full(len(rows), np.nan)
        for train_c, test_c in folds:
            test_rows = np.isin(course_id, test_c)
            gp = train_kinetic_gp(kin.iloc[train_c])
            mean, _ = predict_ln_k(gp, rows.loc[test_rows, DESCRIPTORS])
            gp_pred[test_rows] = conversion(rows.loc[test_rows, 'time_min'].to_numpy(float), np.exp(mean), order) * 100
            gbm = train_direct_gbm(rows[np.isin(course_id, train_c)])
            gbm_pred[test_rows] = np.clip(gbm.predict(rows.loc[test_rows, DESCRIPTORS + ['time_min']].to_numpy()), 0, 100)
        for name, pred in (('Kinetic GP', gp_pred), ('Direct GBM', gbm_pred)):
            err = pred - rows.conversion_percent.to_numpy()
            ss_tot = ((rows.conversion_percent - rows.conversion_percent.mean()) ** 2).sum()
            records.append({'Validation': scheme, 'Model': name, 'RMSE (pp)': np.sqrt(np.mean(err ** 2)),
                            'MAE (pp)': np.mean(np.abs(err)), 'R²': 1 - (err ** 2).sum() / ss_tot})
            preds.append(pd.DataFrame({'Validation': scheme, 'Model': name, 'observed': rows.conversion_percent,
                                       'predicted': pred, 'aldehyde_R_group': rows.aldehyde_R_group,
                                       'amine_name': rows.amine_name}))
    return pd.DataFrame(records), pd.concat(preds, ignore_index=True)
