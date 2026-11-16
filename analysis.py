"""Kinetic and linear free-energy analysis for the imine dashboard (numpy only).

Kinetic model, written in fractional conversion x of the limiting reagent:
    dx/dt = k_app (1 - x)^n,  x(0) = 0
n = 1 is first order; n = 2 is second order with equimolar reagents. For n != 1,
k_app = k * c0^(n-1), where c0 is the initial concentration, so k_app has units of 1/time.
"""
import numpy as np
import pandas as pd

R_GAS = 8.314462618        # J mol^-1 K^-1
KB_OVER_H = 2.083661912e10  # Boltzmann / Planck, K^-1 s^-1
SERIES_KEYS = ['aldehyde_name', 'aldehyde_R_group', 'amine_name', 'solvent', 'temperature_C']


def conversion(t, k, n):
    """Fractional conversion at time t for rate constant k_app and order n."""
    t = np.asarray(t, dtype=float)
    if abs(n - 1) < 1e-9:
        return 1 - np.exp(-k * t)
    base = np.clip(1 + (n - 1) * k * t, 1e-12, None)
    return 1 - base ** (1 / (1 - n))


def _jacobian(t, k, n):
    if abs(n - 1) < 1e-9:
        return t * np.exp(-k * t)
    base = np.clip(1 + (n - 1) * k * t, 1e-12, None)
    return t * base ** (n / (1 - n))


def linearised(x, n):
    """Integrated rate law g(x) = k_app * t, so g(x) against t is a straight line through the origin."""
    x = np.clip(np.asarray(x, dtype=float), 0, 1 - 1e-9)
    if abs(n - 1) < 1e-9:
        return -np.log(1 - x)
    return ((1 - x) ** (1 - n) - 1) / (n - 1)


def time_to_reach(x_target, k, n):
    return linearised(x_target, n) / k


def _series_arrays(data):
    """Pad every (aldehyde, amine, solvent, T) time course into S x m arrays of t and fractional x."""
    frame = data.sort_values('time_min')
    groups = frame.groupby(SERIES_KEYS, sort=True)
    index = pd.DataFrame(list(groups.groups.keys()), columns=SERIES_KEYS)
    width = int(groups.size().max())
    t = np.full((len(index), width), np.nan)
    x = np.full((len(index), width), np.nan)
    for row, (_, part) in enumerate(groups):
        t[row, :len(part)] = part.time_min.to_numpy(float)
        x[row, :len(part)] = part.conversion_percent.to_numpy(float) / 100
    return index, t, x


def _fit_k(t, x, n, iterations=40):
    """Least-squares k_app for each series (rows) at fixed order n, by Gauss-Newton on conversion."""
    mask = ~np.isnan(t)
    tz, xz = np.where(mask, t, 0.0), np.where(mask, x, 0.0)
    g = np.where(mask, linearised(xz, n), 0.0)
    k = np.clip((g * tz).sum(1) / (tz ** 2).sum(1), 1e-9, None)  # straight-line estimate as the starting guess
    for _ in range(iterations):
        kk = k[:, None]
        resid = np.where(mask, xz - conversion(tz, kk, n), 0.0)
        jac = np.where(mask, _jacobian(tz, kk, n), 0.0)
        step = (jac * resid).sum(1) / np.clip((jac ** 2).sum(1), 1e-300, None)
        k = np.where(k + step > 0, k + step, k * 0.2)
    kk = k[:, None]
    resid = np.where(mask, xz - conversion(tz, kk, n), 0.0)
    jac = np.where(mask, _jacobian(tz, kk, n), 0.0)
    m = mask.sum(1)
    sse = (resid ** 2).sum(1)
    se = np.sqrt(sse / np.clip(m - 1, 1, None) / np.clip((jac ** 2).sum(1), 1e-300, None))
    return k, se, sse, m


def fit_kinetics(data, n):
    """One row per series: k_app (1/min), its standard error, RMSE (percentage points) and point count."""
    index, t, x = _series_arrays(data)
    k, se, sse, m = _fit_k(t, x, n)
    out = index.copy()
    out['k_app_per_min'] = k
    out['k_app_se'] = se
    out['rmse_pp'] = np.sqrt(sse / m) * 100
    out['n_points'] = m
    out['sse'] = sse
    return out


def order_profile(data, orders=None):
    """Total SSE over all series as a function of a shared reaction order n."""
    orders = np.round(np.arange(0.5, 3.0001, 0.02), 2) if orders is None else orders
    _, t, x = _series_arrays(data)
    return pd.DataFrame({'n': orders, 'sse': [_fit_k(t, x, n)[2].sum() for n in orders]})


def model_comparison(data):
    """Compare first order, second order and a globally fitted order n by RMSE and AIC."""
    coarse = order_profile(data)
    centre = float(coarse.loc[coarse.sse.idxmin(), 'n'])
    fine = order_profile(data, np.round(np.arange(max(0.5, centre - 0.1), centre + 0.1001, 0.002), 3))
    profile = pd.concat([coarse, fine]).drop_duplicates('n').sort_values('n').reset_index(drop=True)
    best = profile.loc[profile.sse.idxmin()]
    _, t, x = _series_arrays(data)
    n_obs = int((~np.isnan(t)).sum())
    n_series = t.shape[0]
    # Approximate 95% interval for n: orders whose SSE stays within the F-test threshold of the minimum.
    limit = best.sse * (1 + 3.84 / max(n_obs - n_series - 1, 1))
    inside = profile[profile.sse <= limit].n
    rows = []
    for name, order, extra in (('First order', 1.0, 0), ('Second order (equimolar)', 2.0, 0),
                               ('Global order n (fitted)', float(best.n), 1)):
        sse = _fit_k(t, x, order)[2].sum()
        params = n_series + extra
        rows.append({'Model': name, 'Order n': order, 'Parameters': params,
                     'RMSE (pp)': np.sqrt(sse / n_obs) * 100,
                     'AIC': n_obs * np.log(sse / n_obs) + 2 * params})
    table = pd.DataFrame(rows)
    table['ΔAIC'] = table.AIC - table.AIC.min()
    return table, profile, (float(inside.min()), float(inside.max()))


def ols(x, y):
    """Ordinary least squares with standard errors. Returns None when there are fewer than 2 points."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 2 or np.ptp(x) == 0:
        return None
    sxx = ((x - x.mean()) ** 2).sum()
    slope = ((x - x.mean()) * (y - y.mean())).sum() / sxx
    intercept = y.mean() - slope * x.mean()
    resid = y - (intercept + slope * x)
    sse = (resid ** 2).sum()
    sst = ((y - y.mean()) ** 2).sum()
    s2 = sse / (n - 2) if n > 2 else np.nan
    return {'slope': slope, 'intercept': intercept,
            'se_slope': np.sqrt(s2 / sxx) if n > 2 else np.nan,
            'se_intercept': np.sqrt(s2 * (1 / n + x.mean() ** 2 / sxx)) if n > 2 else np.nan,
            'r2': 1 - sse / sst if sst > 0 else np.nan, 'n': n}


def activation_parameters(kin, k_column='k_per_s'):
    """Arrhenius and Eyring parameters for each (aldehyde, amine, solvent) across temperatures."""
    rows = []
    for keys, part in kin.groupby(['aldehyde_name', 'aldehyde_R_group', 'amine_name', 'solvent']):
        temp_k = part.temperature_C.to_numpy(float) + 273.15
        k = part[k_column].to_numpy(float)
        arr = ols(1 / temp_k, np.log(k))
        eyr = ols(1 / temp_k, np.log(k / temp_k))
        if arr is None or eyr is None:
            continue
        dh = -eyr['slope'] * R_GAS / 1000
        ds = (eyr['intercept'] - np.log(KB_OVER_H)) * R_GAS
        rows.append(dict(zip(['aldehyde_name', 'aldehyde_R_group', 'amine_name', 'solvent'], keys),
                         Ea_kJ=-arr['slope'] * R_GAS / 1000, Ea_se=arr['se_slope'] * R_GAS / 1000,
                         lnA=arr['intercept'], dH_kJ=dh, dH_se=eyr['se_slope'] * R_GAS / 1000,
                         dS_J=ds, dS_se=eyr['se_intercept'] * R_GAS,
                         dG298_kJ=dh - 298.15 * ds / 1000, r2=arr['r2'], n_temps=arr['n'],
                         T_min=temp_k.min() - 273.15, T_max=temp_k.max() - 273.15))
    return pd.DataFrame(rows)
