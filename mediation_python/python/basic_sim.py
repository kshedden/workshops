# ## Basic mediation analysis with simulated data

import numpy as np
import statsmodels.api as sm
import pandas as pd

# Make the simulation reproducible

np.random.seed(2343)

# The sample size

n = 400

# The function below simulates data that exhibits different types of
# mediation behavior.  The type of mediation behavior is controlled by
# the 'mode' argument.

def gendat(mode):
    """
    Generate data for demonstrating a mediation analysis.  Setting
    mode = 0, 1, 2, correspond, respectively, to no, full, and partial
    mediation, respectively.
    """

    # The exposure
    x = np.random.normal(size=n)

    # The mediator
    m = x + np.random.normal(size=n)
    m /= np.sqrt(2)

    if mode == 0:
        # No mediation
        y = x + np.random.normal(size=n)
    elif mode == 1:
        # Full mediation
        y = m + np.random.normal(size=n)
    else:
        # Partial mediation
        y = m + x + np.random.normal(size=n)

    return pd.DataFrame({"x": x, "m": m, "y": y})

# The function below carries out a simplified mediation analysis.  The
# purpose of this analysis is to illustrate the main idea behind how
# estimates of mediation are constructed.  It omits a few important
# but technical steps for the sake of clarity.

def fake_mediation(mode):
    """
    Conduct a simplified mediation analysis.  This shows the most
    important steps, but is incomplete since is treats the fitted
    models as being the exactly equal to the population.
    """

    df = gendat(mode)
    m_model = sm.OLS.from_formula("m ~ x", data=df).fit()
    o_model = sm.OLS.from_formula("y ~ x + m", data=df).fit()

    # Create counterfactual mediator values, forcing the exposure
    # to be low.
    df_xlow = df.copy()
    df_xlow.x = 0
    m_xlow = m_model.predict(exog=df_xlow)
    m_xlow += np.sqrt(m_model.scale) * np.random.normal(size=n)

    # Create counterfactual mediator values, forcing the exposure
    # to be high.
    df_xhigh = df.copy()
    df_xhigh.x = 1
    m_xhigh = m_model.predict(exog=df_xhigh)
    m_xhigh += np.sqrt(m_model.scale) * np.random.normal(size=n)

    # Create counterfactual outcomes for the indirect effect.
    df0 = df.copy()
    df0["x"] = 0
    df0["m"] = m_xlow
    y_low = o_model.predict(exog=df0)
    y_low += np.sqrt(o_model.scale) * np.random.normal(size=n)
    df0["x"] = 0
    df0["m"] = m_xhigh
    y_high = o_model.predict(exog=df0)
    y_high += np.sqrt(o_model.scale) * np.random.normal(size=n)

    # The average indirect effect
    aie = np.mean(y_high - y_low)
    aie_se = np.std(y_high - y_low) / np.sqrt(n)

    # Create counterfactual outcomes for the direct effect.
    df0 = df.copy()
    df0["x"] = 0
    df0["m"] = m_xlow
    y_low = o_model.predict(exog=df0)
    y_low += np.sqrt(o_model.scale) * np.random.normal(size=n)
    df0["x"] = 1
    y_high = o_model.predict(exog=df0)
    y_high += np.sqrt(o_model.scale) * np.random.normal(size=n)

    # The average direct effect
    ade = np.mean(y_high - y_low)
    ade_se = np.std(y_high - y_low) / np.sqrt(n)

    return aie, aie_se, ade, ade_se


# Run the simplified mediation analysis for each type of mediation (no
# mediation, full mediation, partial mediation).

for mode in 0, 1, 2:
    aie, aie_se, ade, ade_se = fake_mediation(mode)
    print("AIE=%8.4f (%.4f)  ADE=%8.4f (%.4f)" % (aie, aie_se, ade, ade_se))


# Run a mediation analysis using the Mediation package for each type
# of mediation (no/full/partial).

for mode in 0, 1, 2:

    df = gendat(mode)
    outcome_model = sm.OLS.from_formula("y ~ x + m", data=df)
    mediator_model = sm.OLS.from_formula("m ~ x", data=df)
    med = sm.stats.Mediation(outcome_model, mediator_model, "x", "m").fit(n_rep=100)
    print(med.summary(), "\n")
