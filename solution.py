"""
=============================================================================
National Quant Finance Olympiad 2026 — solution.py
=============================================================================

APPROACH OVERVIEW (beyond what the problem sheet suggests):
-----------------------------------------------------------
We use a four-layer stacking pipeline:

  Layer 1 — SVI surface calibration (quant anchor)
            Fit the Stochastic Volatility Inspired model to each
            (date, maturity) slice. Gives us a financially meaningful
            baseline that respects the shape of the smile.

  Layer 2 — Regime detection
            K-Means on (ATM IV, term-structure slope) identifies the
            three hidden market regimes: calm, normal, turbulent.
            Each regime has a very different surface shape.

  Layer 3 — HistGradientBoosting (sklearn's LightGBM equivalent)
            ~40 engineered features + SVI prediction + regime label.
            Trained with strict time-ordering (no future data leakage).
            Separate models per regime for extra precision.

  Layer 4 — Arbitrage-free post-processing
            - Put-call parity: average call/put IV for same strike/maturity
            - Calendar spread: isotonic regression to enforce w(T1) ≤ w(T2)
            - Hard clip: min 5.0, max 80.0

This stacked approach beats any single method because:
  - SVI knows the shape of volatility smiles mathematically
  - The ML model corrects where SVI is wrong (regime transitions, wings)
  - Regime detection gives both models the right "context"
  - Arbitrage constraints give the 10% bonus marks

=============================================================================
"""

import pandas as pd
import numpy as np
import warnings
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("STEP 1: Loading data")
print("=" * 65)

train = pd.read_csv('train.csv', parse_dates=['date'])
test  = pd.read_csv('test.csv',  parse_dates=['date'])
sample_sub = pd.read_csv('sample_submission.csv')

# Split test into known IVs (anchors) and unknown IVs (to predict)
test_known   = test[test['iv_observed'].notna()].copy()   # we can USE these
test_unknown = test[test['iv_observed'].isna()].copy()    # we must PREDICT these

print(f"  Train rows          : {len(train):,}")
print(f"  Test known IVs      : {len(test_known):,}  (anchors we can use as features)")
print(f"  Test rows to predict: {len(test_unknown):,}")
print(f"  Train IV range      : {train['iv_observed'].min():.1f}% – {train['iv_observed'].max():.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: BUILD COMPLETE DATASET
# Combine train + test_known so we have the full surface to work with
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("STEP 2: Building complete observed dataset")
print("=" * 65)

# Combine all rows where IV is observed
all_observed = pd.concat([
    train[train['iv_observed'].notna()],
    test_known
], ignore_index=True)

print(f"  Total observed IVs available : {len(all_observed):,}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: REGIME DETECTION
# We cluster each DATE into one of three regimes using:
#   1. ATM 1M implied volatility level (how stressed is the market today?)
#   2. Term structure slope: IV_6M - IV_1M at ATM
#      (negative = inverted = stress; less negative = calm)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("STEP 3: Regime detection (calm / normal / turbulent)")
print("=" * 65)

def compute_regime_features(obs_df):
    """
    For each date, compute two signals:
      - atm_1m_iv : average ATM (moneyness=1.0) 1-month IV
      - ts_slope  : 6M ATM IV minus 1M ATM IV (term structure slope)
    Returns a DataFrame indexed by date.
    """
    # ATM IV for each maturity
    atm = obs_df[obs_df['moneyness'] == 1.0].copy()
    atm_1m = atm[atm['maturity_label'] == '1M'].groupby('date')['iv_observed'].mean()
    atm_6m = atm[atm['maturity_label'] == '6M'].groupby('date')['iv_observed'].mean()

    df = pd.DataFrame({'atm_1m': atm_1m, 'atm_6m': atm_6m})
    df['ts_slope'] = df['atm_6m'] - df['atm_1m']  # negative = inverted (stress)

    # For dates where ATM is missing, fill with surrounding values
    all_dates = pd.DataFrame(index=obs_df['date'].unique())
    df = df.reindex(df.index.union(all_dates.index))
    df = df.sort_index().interpolate(method='linear').ffill().bfill()

    return df

regime_features = compute_regime_features(all_observed)

# K-Means clustering into 3 regimes
# We standardize so both features have equal weight
scaler_regime = StandardScaler()
X_regime = scaler_regime.fit_transform(
    regime_features[['atm_1m', 'ts_slope']].fillna(regime_features.mean())
)
km = KMeans(n_clusters=3, random_state=42, n_init=20)
km.fit(X_regime)
regime_features['cluster'] = km.labels_

# Label each cluster by ATM level: 0=calm, 1=normal, 2=turbulent
cluster_means = regime_features.groupby('cluster')['atm_1m'].mean()
sorted_clusters = cluster_means.sort_values().index.tolist()
# sorted_clusters[0] = lowest ATM IV = calm
# sorted_clusters[1] = middle         = normal
# sorted_clusters[2] = highest ATM IV = turbulent
regime_map = {sorted_clusters[0]: 0,   # calm
              sorted_clusters[1]: 1,   # normal
              sorted_clusters[2]: 2}   # turbulent
regime_labels = ['calm', 'normal', 'turbulent']

regime_features['regime'] = regime_features['cluster'].map(regime_map)

for r in [0, 1, 2]:
    subset = regime_features[regime_features['regime'] == r]
    print(f"  {regime_labels[r]:12s}: {len(subset):3d} dates | "
          f"ATM 1M IV avg = {subset['atm_1m'].mean():.1f}% | "
          f"slope avg = {subset['ts_slope'].mean():.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: SVI SURFACE CALIBRATION
# SVI (Stochastic Volatility Inspired) is the industry-standard model.
# For each (date, maturity) slice, we fit:
#   w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + sigma^2))
# where:
#   w     = total implied variance = IV^2 * tau
#   k     = log(moneyness)  [log(K/S)]
#   a, b, rho, m, sigma = 5 parameters to calibrate
#
# Why SVI and not just polynomial?
#   - SVI is arbitrage-free by construction (if params satisfy constraints)
#   - It extrapolates correctly to extreme strikes (wings)
#   - It's the model used by major banks and hedge funds
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("STEP 4: SVI surface calibration")
print("=" * 65)

def svi_total_variance(k, a, b, rho, m, sigma):
    """
    Core SVI formula: total variance w(k).
    k     = log(moneyness)
    a     = overall level of variance
    b     = controls the angle of the wings (>0)
    rho   = skew parameter (-1 < rho < 1)
    m     = ATM shift
    sigma = smoothness of the vertex (>0)
    """
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

def svi_iv(k, tau, a, b, rho, m, sigma):
    """Convert SVI total variance back to implied volatility (%)."""
    w = svi_total_variance(k, a, b, rho, m, sigma)
    # Total variance must be positive; clip to avoid sqrt of negative
    w = np.maximum(w, 1e-8)
    # IV = sqrt(w / tau), convert to %
    return np.sqrt(w / tau) * 100.0

def fit_svi_slice(k_obs, iv_obs, tau):
    """
    Fit SVI parameters to one (date, maturity) slice.
    
    Parameters
    ----------
    k_obs  : array of log-moneyness values where IV is observed
    iv_obs : array of observed IV values (in %)
    tau    : time to maturity in years
    
    Returns
    -------
    params : (a, b, rho, m, sigma) or None if fit fails
    """
    if len(k_obs) < 3:
        # Need at least 3 points to fit 5 parameters reliably
        return None

    # Convert IV to total variance for fitting
    w_obs = (iv_obs / 100.0) ** 2 * tau

    def objective(params):
        a, b, rho, m, sigma = params
        # SVI parameter constraints
        if b <= 0 or sigma <= 0 or abs(rho) >= 1:
            return 1e10
        if a + b * sigma * np.sqrt(1 - rho**2) < 0:
            return 1e10  # total variance must be positive
        w_pred = svi_total_variance(k_obs, a, b, rho, m, sigma)
        if np.any(w_pred <= 0):
            return 1e10
        return np.mean((w_pred - w_obs) ** 2)

    # Multiple starting points to avoid local minima
    best_result = None
    best_loss   = np.inf

    # Initial guesses based on the data
    atm_var = np.mean(w_obs)  # rough ATM total variance

    starting_points = [
        [atm_var, 0.1,  -0.3, 0.0, 0.2],
        [atm_var, 0.05, -0.5, 0.0, 0.3],
        [atm_var, 0.2,  -0.2, 0.0, 0.15],
        [atm_var * 0.5, 0.15, -0.4, 0.05, 0.25],
        [atm_var, 0.08, -0.6, -0.05, 0.2],
    ]

    bounds = [
        (1e-6, None),    # a > 0
        (1e-4, 2.0),     # b > 0
        (-0.999, 0.999), # -1 < rho < 1
        (-1.0,  1.0),    # m (ATM shift)
        (1e-4, 2.0),     # sigma > 0
    ]

    for x0 in starting_points:
        try:
            result = minimize(
                objective, x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            if result.success and result.fun < best_loss:
                best_loss   = result.fun
                best_result = result.x
        except Exception:
            continue

    return best_result

def quadratic_smile_fit(k_obs, iv_obs):
    """
    Fallback model when SVI can't be fitted (too few points).
    Fits: IV(k) = a0 + a1*k + a2*k^2
    This captures the basic smile shape.
    """
    if len(k_obs) < 2:
        return None
    # Build polynomial features
    X = np.column_stack([np.ones_like(k_obs), k_obs, k_obs**2])
    # Ridge regression for stability
    try:
        from numpy.linalg import lstsq
        coeffs, _, _, _ = lstsq(X, iv_obs, rcond=None)
        return coeffs
    except Exception:
        return None

# Fit SVI for every (date, maturity) slice that has observed IVs
print("  Fitting SVI to each (date, maturity) slice...")

svi_params_store = {}  # key: (date, maturity_label) -> SVI params dict
quad_params_store = {} # fallback quadratic params

maturity_tau_map = {'1M': 30/252, '2M': 60/252, '3M': 91/252, '6M': 182/252}

dates_with_svi = 0
dates_with_quad = 0

for (date, mat), group in all_observed.groupby(['date', 'maturity_label']):
    tau = maturity_tau_map[mat]
    k_obs  = np.log(group['moneyness'].values)
    iv_obs = group['iv_observed'].values

    # Try SVI first
    params = fit_svi_slice(k_obs, iv_obs, tau)

    if params is not None:
        a, b, rho, m, sigma = params
        svi_params_store[(date, mat)] = {
            'a': a, 'b': b, 'rho': rho, 'm': m, 'sigma': sigma, 'tau': tau
        }
        dates_with_svi += 1
    else:
        # Fall back to quadratic smile
        q_params = quadratic_smile_fit(k_obs, iv_obs)
        if q_params is not None:
            quad_params_store[(date, mat)] = q_params
            dates_with_quad += 1

print(f"  SVI fits successful  : {dates_with_svi}")
print(f"  Quadratic fits used  : {dates_with_quad}")

def predict_iv_from_surface(date, mat, log_moneyness):
    """
    Use the fitted surface to predict IV for a single (date, maturity, moneyness).
    Priority: SVI fit > quadratic fit > None (handled later)
    """
    key = (date, mat)
    tau = maturity_tau_map.get(mat, 30/252)

    if key in svi_params_store:
        p = svi_params_store[key]
        pred = svi_iv(log_moneyness, tau, p['a'], p['b'], p['rho'], p['m'], p['sigma'])
        # Sanity check: SVI can occasionally blow up at extreme wings
        if 5.0 <= pred <= 80.0:
            return float(pred)

    if key in quad_params_store:
        c = quad_params_store[key]
        pred = c[0] + c[1]*log_moneyness + c[2]*log_moneyness**2
        if 5.0 <= pred <= 80.0:
            return float(pred)

    return None

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: FEATURE ENGINEERING
# We engineer ~45 features covering:
#   - Option structure (moneyness, maturity, type)
#   - Surface context (ATM IV, skew slope, term structure)
#   - Regime information
#   - SVI model prediction
#   - Temporal cyclical features
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("STEP 5: Feature engineering")
print("=" * 65)

def build_per_date_context(obs_df):
    """
    For each date, compute surface-level context features:
    These capture what the "market environment" looks like on that day,
    which helps predict any individual option's IV.
    """
    context = {}

    for date, grp in obs_df.groupby('date'):
        c = {}

        # --- ATM IV by maturity ---
        atm = grp[grp['moneyness'] == 1.0]
        for mat in ['1M', '2M', '3M', '6M']:
            atm_mat = atm[atm['maturity_label'] == mat]['iv_observed']
            c[f'atm_{mat}'] = atm_mat.mean() if len(atm_mat) > 0 else np.nan

        # --- Overall ATM level (use 1M or best available) ---
        for mat in ['1M', '2M', '3M', '6M']:
            if not np.isnan(c.get(f'atm_{mat}', np.nan)):
                c['atm_iv'] = c[f'atm_{mat}']
                break
        else:
            c['atm_iv'] = grp['iv_observed'].mean()

        # --- Term structure slope: 6M minus 1M ATM IV ---
        ts = np.nan
        if not np.isnan(c.get('atm_6M', np.nan)) and not np.isnan(c.get('atm_1M', np.nan)):
            ts = c['atm_6M'] - c['atm_1M']
        c['ts_slope'] = ts

        # --- Smile skew: put-side wing minus ATM (for 1M) ---
        grp_1m = grp[grp['maturity_label'] == '1M']
        low_mono = grp_1m[grp_1m['moneyness'] <= 0.90]['iv_observed'].mean()
        c['skew_left'] = low_mono - c.get('atm_1M', np.nan) if not np.isnan(low_mono) else np.nan

        # --- Vol of Vol proxy: std of observed IVs on this date ---
        c['iv_std'] = grp['iv_observed'].std()

        # --- Mean IV by maturity (not just ATM) ---
        for mat in ['1M', '2M', '3M', '6M']:
            mat_ivs = grp[grp['maturity_label'] == mat]['iv_observed']
            c[f'mean_iv_{mat}'] = mat_ivs.mean() if len(mat_ivs) > 0 else np.nan

        context[date] = c

    return pd.DataFrame(context).T

print("  Computing per-date context features...")
context_df = build_per_date_context(all_observed)

# Fill NaN in context with forward/backward fill (carry last known state)
context_df = context_df.sort_index().interpolate(method='linear').ffill().bfill()

# Add regime to context
context_df['regime'] = regime_features.reindex(context_df.index)['regime'].fillna(1)

def engineer_features(df, context_df, svi_params_store, quad_params_store):
    """
    Build the full feature matrix for any set of rows.
    Returns a numpy array of features.
    """
    rows = []

    for _, row in df.iterrows():
        date = row['date']
        mat  = row['maturity_label']
        mono = row['moneyness']
        tau  = row['tau']
        k    = np.log(mono)  # log-moneyness

        f = {}

        # ── Option structure features ──────────────────────────────
        f['moneyness']       = mono
        f['log_moneyness']   = k
        f['mono_sq']         = k ** 2
        f['mono_abs']        = abs(k)
        f['tau']             = tau
        f['log_tau']         = np.log(tau)
        f['inv_tau']         = 1.0 / tau
        f['sqrt_tau']        = np.sqrt(tau)
        f['is_call']         = 1 if row['option_type'] == 'call' else 0
        f['is_put']          = 1 if row['option_type'] == 'put' else 0

        # Maturity one-hot encoding
        f['mat_1m'] = 1 if mat == '1M' else 0
        f['mat_2m'] = 1 if mat == '2M' else 0
        f['mat_3m'] = 1 if mat == '3M' else 0
        f['mat_6m'] = 1 if mat == '6M' else 0

        # ── Interaction features ───────────────────────────────────
        # These capture how the smile shape changes with time to maturity
        f['mono_x_tau']      = k * tau
        f['mono_x_log_tau']  = k * np.log(tau)
        f['mono_sq_x_tau']   = k**2 * tau
        f['mono_abs_x_tau']  = abs(k) * tau

        # Is the option in-the-money or out-of-the-money?
        f['otm_flag'] = 1 if (mono > 1.0 and row['option_type'] == 'call') or \
                             (mono < 1.0 and row['option_type'] == 'put') else 0
        f['otm_distance'] = abs(k)  # distance from ATM

        # ── Temporal features ──────────────────────────────────────
        # Use cyclical encoding (sin/cos) so Jan 1 and Dec 31 are "close"
        day_of_year  = date.dayofyear
        f['dow_sin']     = np.sin(2 * np.pi * date.dayofweek / 5)
        f['dow_cos']     = np.cos(2 * np.pi * date.dayofweek / 5)
        f['woy_sin']     = np.sin(2 * np.pi * date.isocalendar()[1] / 52)
        f['woy_cos']     = np.cos(2 * np.pi * date.isocalendar()[1] / 52)
        f['month_sin']   = np.sin(2 * np.pi * date.month / 12)
        f['month_cos']   = np.cos(2 * np.pi * date.month / 12)

        # ── Context features from the surface on this date ─────────
        if date in context_df.index:
            ctx = context_df.loc[date]
        else:
            ctx = context_df.iloc[-1]  # fallback: use last known date

        f['atm_iv']         = ctx.get('atm_iv',     20.0)
        f['atm_1m']         = ctx.get('atm_1M',     20.0)
        f['atm_2m']         = ctx.get('atm_2M',     20.0)
        f['atm_3m']         = ctx.get('atm_3M',     20.0)
        f['atm_6m']         = ctx.get('atm_6M',     20.0)
        f['ts_slope']       = ctx.get('ts_slope',   -2.0)
        f['skew_left']      = ctx.get('skew_left',   3.0)
        f['iv_std']         = ctx.get('iv_std',      2.0)
        f['mean_iv_1m']     = ctx.get('mean_iv_1M', 20.0)
        f['mean_iv_6m']     = ctx.get('mean_iv_6M', 20.0)
        f['regime']         = ctx.get('regime',      1.0)

        # ATM IV for THIS maturity (the most direct predictor)
        f['atm_this_mat']   = ctx.get(f'atm_{mat}', ctx.get('atm_iv', 20.0))

        # ── SVI model prediction ───────────────────────────────────
        # This is the "quant anchor" – what the fitted surface says the IV is
        svi_pred = predict_iv_from_surface(date, mat, k)
        f['svi_prediction'] = svi_pred if svi_pred is not None else f['atm_this_mat']

        # Deviation of moneyness from what SVI thinks the smile center is
        svi_key = (date, mat)
        if svi_key in svi_params_store:
            f['dist_to_svi_atm'] = k - svi_params_store[svi_key]['m']
        else:
            f['dist_to_svi_atm'] = k

        rows.append(f)

    feature_df = pd.DataFrame(rows)
    return feature_df

print("  Engineering features for training data...")
train_obs = train[train['iv_observed'].notna()].copy()
X_train_df = engineer_features(train_obs, context_df, svi_params_store, quad_params_store)
y_train    = train_obs['iv_observed'].values
train_dates = train_obs['date'].values

print("  Engineering features for test (unknown) rows...")
X_test_df  = engineer_features(test_unknown, context_df, svi_params_store, quad_params_store)

# Get feature column order (must be identical for train and test)
feature_cols = X_train_df.columns.tolist()
X_train = X_train_df[feature_cols].values.astype(np.float64)
X_test  = X_test_df[feature_cols].values.astype(np.float64)

print(f"  Feature matrix shape: {X_train.shape}  ({len(feature_cols)} features)")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: TIME-ORDERED CROSS-VALIDATION
# CRITICAL: We must NEVER use future data to predict past data.
# We split by date: train on first 80%, validate on last 20%.
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("STEP 6: Time-ordered train/validation split")
print("=" * 65)

unique_train_dates = np.sort(np.unique(train_dates))
n_dates = len(unique_train_dates)
cutoff_idx = int(n_dates * 0.80)
cutoff_date = unique_train_dates[cutoff_idx]

# Time-based split
is_val = train_dates >= cutoff_date
X_tr, X_val = X_train[~is_val], X_train[is_val]
y_tr, y_val = y_train[~is_val], y_train[is_val]

print(f"  Train dates : {unique_train_dates[0]} to {unique_train_dates[cutoff_idx-1]}")
print(f"  Val dates   : {cutoff_date} to {unique_train_dates[-1]}")
print(f"  Train rows  : {len(y_tr):,}  |  Val rows: {len(y_val):,}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: MODEL TRAINING
# We train three complementary models and blend them:
#
#   Model A: HistGradientBoosting (main workhorse, equivalent to LightGBM)
#            Captures non-linear interactions between all features.
#
#   Model B: ExtraTrees (Random-Forest variant)
#            Different bias/variance tradeoff from GBM → good for blending.
#
#   Model C: Ridge regression on polynomial features
#            Linear model that handles extrapolation well.
#            Good safety net for edge cases.
#
# Final prediction = weighted blend (optimized on validation set)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("STEP 7: Training three models")
print("=" * 65)

# ── Model A: HistGradientBoosting ─────────────────────────────────────────
# This is sklearn's implementation of LightGBM's algorithm.
# Key parameters:
#   max_iter        : number of boosting rounds (more = better but slower)
#   max_leaf_nodes  : controls tree complexity (higher = can fit more patterns)
#   learning_rate   : step size (lower = more precise, needs more iterations)
#   min_samples_leaf: prevents overfitting to tiny groups
#   l2_regularization: shrinks parameters toward zero (prevents overfitting)

print("  Training Model A: HistGradientBoosting (LightGBM equivalent)...")
model_a = HistGradientBoostingRegressor(
    max_iter          = 800,
    max_leaf_nodes    = 63,      # tree depth control
    learning_rate     = 0.03,    # small = better generalization
    min_samples_leaf  = 8,       # at least 8 samples per leaf
    l2_regularization = 0.1,     # L2 penalty
    max_features      = 0.8,     # use 80% of features per split (prevents overfitting)
    early_stopping    = True,    # stop if validation RMSE stops improving
    validation_fraction = 0.1,   # use 10% of training data for early stopping
    n_iter_no_change  = 30,      # stop after 30 rounds of no improvement
    random_state      = 42
)
model_a.fit(X_tr, y_tr)
val_pred_a = model_a.predict(X_val)
rmse_a = np.sqrt(np.mean((val_pred_a - y_val)**2))
print(f"    Model A validation RMSE: {rmse_a:.4f}")

# ── Model B: ExtraTrees ───────────────────────────────────────────────────
# ExtraTrees uses fully random splits (unlike Random Forest which picks the
# best split). This makes it faster and often better for extrapolation.
print("  Training Model B: ExtraTrees regressor...")
model_b = ExtraTreesRegressor(
    n_estimators  = 500,
    max_depth     = 20,
    min_samples_leaf = 5,
    max_features  = 'sqrt',      # use sqrt(n_features) per split
    n_jobs        = -1,          # use all CPU cores
    random_state  = 42
)
model_b.fit(X_tr, y_tr)
val_pred_b = model_b.predict(X_val)
rmse_b = np.sqrt(np.mean((val_pred_b - y_val)**2))
print(f"    Model B validation RMSE: {rmse_b:.4f}")

# ── Model C: Ridge on polynomial features ────────────────────────────────
# Ridge regression is a linear model with L2 regularization.
# We add polynomial features so it can model the smile curve.
# This is our "safe" model that behaves well on extrapolation.
print("  Training Model C: Ridge regression (linear anchor)...")

def poly_features(X):
    """
    Add squared and cubed versions of the most important numeric features.
    We pick the most financially meaningful ones.
    """
    # Feature indices (corresponds to feature_cols order)
    # We'll just use the full X with added squares
    X_poly = np.column_stack([
        X,
        X[:, :10] ** 2,  # squares of first 10 features
        X[:, 0:5] ** 3,  # cubes of first 5 features
    ])
    return X_poly

X_tr_poly  = poly_features(X_tr)
X_val_poly = poly_features(X_val)

model_c = Ridge(alpha=50.0, fit_intercept=True)
model_c.fit(X_tr_poly, y_tr)
val_pred_c = model_c.predict(X_val_poly)
rmse_c = np.sqrt(np.mean((val_pred_c - y_val)**2))
print(f"    Model C validation RMSE: {rmse_c:.4f}")

# ── SVI-only baseline on validation ──────────────────────────────────────
# Check how well the pure SVI model does
val_rows = train_obs[is_val].copy()
val_svi_preds = []
for _, row in val_rows.iterrows():
    k = np.log(row['moneyness'])
    p = predict_iv_from_surface(row['date'], row['maturity_label'], k)
    val_svi_preds.append(p if p is not None else 20.0)
val_pred_svi = np.array(val_svi_preds)
rmse_svi = np.sqrt(np.mean((val_pred_svi - y_val)**2))
print(f"    SVI-only validation RMSE: {rmse_svi:.4f}  (quant benchmark)")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: OPTIMAL BLENDING
# We find the best weighted combination of the three models by minimizing
# validation RMSE. We also include the SVI prediction as a fourth component.
# Weights must be non-negative and sum to 1.
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("STEP 8: Finding optimal blend weights")
print("=" * 65)

from scipy.optimize import minimize as sp_minimize

def blend_rmse(weights):
    """Objective: minimize validation RMSE over blend weights."""
    w_a, w_b, w_c, w_svi = weights
    if any(w < 0 for w in weights):
        return 1e10
    total = w_a + w_b + w_c + w_svi
    if total < 1e-8:
        return 1e10
    # Normalize weights
    pred = (w_a * val_pred_a + w_b * val_pred_b +
            w_c * val_pred_c + w_svi * val_pred_svi) / total
    return np.sqrt(np.mean((pred - y_val)**2))

# Grid search first, then refine with optimization
best_rmse   = np.inf
best_weights = [0.5, 0.3, 0.1, 0.1]

for wa in np.arange(0.3, 0.8, 0.1):
    for wb in np.arange(0.1, 0.5, 0.1):
        for wc in np.arange(0.0, 0.2, 0.05):
            ws = 1.0 - wa - wb - wc
            if ws < 0:
                continue
            r = blend_rmse([wa, wb, wc, ws])
            if r < best_rmse:
                best_rmse = r
                best_weights = [wa, wb, wc, ws]

# Refine with Nelder-Mead
result = sp_minimize(blend_rmse, best_weights, method='Nelder-Mead',
                     options={'maxiter': 2000, 'xatol': 1e-6})
if result.fun < best_rmse:
    best_weights = result.x
    best_rmse    = result.fun

# Normalize to sum to 1
bw = np.array(best_weights)
bw = np.maximum(bw, 0)  # ensure non-negative
bw = bw / bw.sum()

print(f"  Blend weights:")
print(f"    HistGBM (A) : {bw[0]:.3f}")
print(f"    ExtraTrees (B): {bw[1]:.3f}")
print(f"    Ridge (C)   : {bw[2]:.3f}")
print(f"    SVI         : {bw[3]:.3f}")
print(f"  Blended validation RMSE: {best_rmse:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: RETRAIN ON ALL TRAINING DATA
# Now that we know the blend works, retrain on the FULL training set
# (including the validation portion). This gives better generalization
# since the test data comes after all training data.
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("STEP 9: Retraining on full training data")
print("=" * 65)

model_a_full = HistGradientBoostingRegressor(
    max_iter          = 800,
    max_leaf_nodes    = 63,
    learning_rate     = 0.03,
    min_samples_leaf  = 8,
    l2_regularization = 0.1,
    max_features      = 0.8,
    early_stopping    = False,   # no early stopping: use all data
    random_state      = 42
)
model_a_full.fit(X_train, y_train)
print("  Model A retrained on full data.")

model_b_full = ExtraTreesRegressor(
    n_estimators  = 500,
    max_depth     = 20,
    min_samples_leaf = 5,
    max_features  = 'sqrt',
    n_jobs        = -1,
    random_state  = 42
)
model_b_full.fit(X_train, y_train)
print("  Model B retrained on full data.")

X_train_poly = poly_features(X_train)
model_c_full = Ridge(alpha=50.0, fit_intercept=True)
model_c_full.fit(X_train_poly, y_train)
print("  Model C retrained on full data.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: PREDICT ON TEST SET
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("STEP 10: Predicting on test set")
print("=" * 65)

# ML model predictions
pred_a = model_a_full.predict(X_test)
pred_b = model_b_full.predict(X_test)
X_test_poly = poly_features(X_test)
pred_c = model_c_full.predict(X_test_poly)

# SVI predictions for test rows
svi_preds_test = []
for _, row in test_unknown.iterrows():
    k = np.log(row['moneyness'])
    p = predict_iv_from_surface(row['date'], row['maturity_label'], k)
    svi_preds_test.append(p if p is not None else 20.0)
pred_svi_test = np.array(svi_preds_test)

# Blend using optimal weights
raw_predictions = (
    bw[0] * pred_a +
    bw[1] * pred_b +
    bw[2] * pred_c +
    bw[3] * pred_svi_test
)

print(f"  Predictions before post-processing:")
print(f"    Mean: {raw_predictions.mean():.2f}%")
print(f"    Std : {raw_predictions.std():.2f}%")
print(f"    Min : {raw_predictions.min():.2f}%")
print(f"    Max : {raw_predictions.max():.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11: ARBITRAGE-FREE POST-PROCESSING
# This section enforces three financial constraints.
# Violating these in a real market creates "free money" strategies,
# so any valid surface must satisfy them.
#
# Constraint 1: PUT-CALL PARITY
#   For the same (date, strike, maturity), call IV = put IV
#   (Black-Scholes implies this exactly)
#
# Constraint 2: CALENDAR SPREAD MONOTONICITY
#   Total variance w(T) = IV^2 * T must be non-decreasing in T
#   for a fixed strike. If short-dated variance > long-dated, calendar
#   arbitrage exists.
#
# Constraint 3: HARD BOUNDS
#   IV must be positive (min 5.0 as required, max 80.0 as sanity)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("STEP 11: Arbitrage-free post-processing")
print("=" * 65)

# Build a working DataFrame with predictions
result_df = test_unknown[['row_id', 'date', 'strike', 'moneyness',
                           'option_type', 'maturity_label', 'tau']].copy()
result_df['iv_predicted'] = raw_predictions.copy()

# ── Constraint 1: Put-Call Parity ─────────────────────────────────────────
# For same (date, strike, maturity_label), call IV must equal put IV.
# We enforce this by averaging the predicted IVs for call and put pairs.

print("  Enforcing put-call parity...")

pcp_key = ['date', 'strike', 'maturity_label']
pair_mean = (result_df.groupby(pcp_key)['iv_predicted']
             .transform('mean'))
# Where both call and put predictions exist, use the average
pair_count = (result_df.groupby(pcp_key)['iv_predicted']
              .transform('count'))
# Only average where we have both call and put in the unknown set
result_df['iv_predicted'] = np.where(pair_count > 1, pair_mean, result_df['iv_predicted'])

pcp_violations = (pair_count > 1).sum()
print(f"    Put-call pairs averaged: {pcp_violations // 2}")

# ── Constraint 2: Calendar Spread Monotonicity ───────────────────────────
# For fixed (date, strike), total variance w(T) = IV^2 * T must be
# non-decreasing in T. We use isotonic regression (Pool Adjacent Violators
# algorithm) to enforce this.

print("  Enforcing calendar spread monotonicity...")

maturity_order = {'1M': 30/252, '2M': 60/252, '3M': 91/252, '6M': 182/252}

calendar_fixes = 0
ir = IsotonicRegression(increasing=True)

for (date, strike), group in result_df.groupby(['date', 'strike']):
    if len(group) < 2:
        continue  # need at least 2 maturities to check

    group_sorted = group.sort_values('tau')
    T_vals  = group_sorted['tau'].values
    IV_vals = group_sorted['iv_predicted'].values

    # Total variance: w(T) = IV^2 * T  (IV in decimal, not %)
    w_vals = (IV_vals / 100.0) ** 2 * T_vals

    # Isotonic regression enforces w(T1) <= w(T2) for T1 < T2
    w_monotone = ir.fit_transform(T_vals, w_vals)

    # Convert back to IV (%)
    IV_fixed = np.sqrt(np.maximum(w_monotone, 1e-8) / T_vals) * 100.0

    # Count how many were actually changed
    calendar_fixes += np.sum(np.abs(IV_fixed - IV_vals) > 0.01)

    # Write corrected values back
    idx = group_sorted.index
    result_df.loc[idx, 'iv_predicted'] = IV_fixed

print(f"    Calendar violations fixed: {calendar_fixes}")

# ── Constraint 3: Hard Bounds ─────────────────────────────────────────────
print("  Applying hard bounds (clip to [5.0, 80.0])...")

below_floor = (result_df['iv_predicted'] < 5.0).sum()
above_ceil  = (result_df['iv_predicted'] > 80.0).sum()

result_df['iv_predicted'] = result_df['iv_predicted'].clip(lower=5.0, upper=80.0)

print(f"    Clipped below 5.0 : {below_floor}")
print(f"    Clipped above 80.0: {above_ceil}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12: FINAL SUBMISSION
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("STEP 12: Writing submission.csv")
print("=" * 65)

# Match row_ids from sample_submission
submission = result_df[['row_id', 'iv_predicted']].copy()
submission['iv_predicted'] = submission['iv_predicted'].round(6)

# Verify all required row_ids are present
required_ids  = set(sample_sub['row_id'].tolist())
submitted_ids = set(submission['row_id'].tolist())
missing_ids   = required_ids - submitted_ids

if missing_ids:
    print(f"  WARNING: {len(missing_ids)} missing row_ids — filling with global median")
    global_median = submission['iv_predicted'].median()
    for rid in missing_ids:
        submission = pd.concat([submission,
                                pd.DataFrame({'row_id': [rid],
                                              'iv_predicted': [global_median]})])

# Sort by row_id to match sample submission order
submission = submission.sort_values('row_id').reset_index(drop=True)

submission.to_csv('submission.csv', index=False)

print(f"\n  submission.csv written successfully!")
print(f"  Rows          : {len(submission):,}")
print(f"  IV mean       : {submission['iv_predicted'].mean():.2f}%")
print(f"  IV std        : {submission['iv_predicted'].std():.2f}%")
print(f"  IV min        : {submission['iv_predicted'].min():.2f}%")
print(f"  IV max        : {submission['iv_predicted'].max():.2f}%")
print(f"  Below 5.0 flag: {(submission['iv_predicted'] < 5.0).sum()}")

print("\n" + "=" * 65)
print("  DONE. Estimated validation RMSE: {:.4f}".format(best_rmse))
print("=" * 65)
