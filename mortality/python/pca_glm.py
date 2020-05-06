import numpy as np
import statsmodels
import statsmodels.base.model as base
from statsmodels.genmod import families
import warnings

class GPCA(base.LikelihoodModel):

    def __init__(self, endog, ndim, offset=None, family=None):
        """
        Fit a generalized principal component analysis.

        This analysis fits a generalized linear model (GLM) to a
        rectangular data array.  The linear predictor, which in a GLM
        would be derived from covariates, is instead represented as a
        factor-structured matrix.  If endog is n x p and we wish to
        extract d factors, then the linear predictor is represented as
        1*icept' + (s - 1*icept')*F*F', where 1 is a column vector of
        n 1's, s is a n x p matrix containing the 'saturated' linear
        predictor, and F is a p x d orthogonal matrix of loadings.

        Parameters
        ----------
        endog : array-like
            The data to which a reduced-rank structure is fit.
        ndim : integer
            The dimension of the low-rank structure.
        family : GLM family instance
            The GLM family to use in the analysis
        offset : array-like
            An optional offset vector

        Returns
        -------
        A GPCAResults instance.

        Notes
        -----
        Estimation uses the Grassmann optimization approach of Edelman,
        rather than the approaches from Landgraf and Lee.

        References
        ----------
        A. Landgraf, Y.Lee (2019). Generalized Principal Component Analysis:
        Projection of saturated model parameters.  Technometrics.
        https://www.asc.ohio-state.edu/lee.2272/mss/tr890.pdf

        Edelman,Arias, Smith (1999).  The geometry of algorithms with orthogonality
        constraints.
        https://arxiv.org/abs/physics/9806030
        """

        if family is None:
            # Default family
            family = families.Gaussian()

        self.family = family
        self.endog = np.asarray(endog)
        self.ndim = ndim

        if offset != None:
            if all(offset.shape != endog.shape):
                msg = "endog and offset must have the same shape"
                raise ValueError(msg)
            offset = self.offset

        # Calculate the saturated parameter
        if isinstance(family, families.Poisson):
            satparam = np.log(endog)
            satparam = np.where(endog == 0, -3, satparam)
        elif isinstance(family, families.Gaussian):
            satparam = endog
        else:
            raise ValueError("Unknown family")
        self.satparam = satparam


    def _linpred(self, params):

        n, p = self.endog.shape

        icept = params[0:p]
        qm = params[p:].reshape((p, self.ndim))

        resid = self.satparam - icept

        lp = icept + np.dot(np.dot(resid, qm), qm.T)

        if hasattr(self, "offset"):
            lp += self.offset

        return icept, qm, resid, lp


    def predict(self, params, linear=False):

        _, _, _, lp = self._linpred(params)

        if linear:
            return lp
        else:
            return self.family.fitted(lp)


    def loglike(self, params):

        _, _, _, lp = self._linpred(params)
        expval = self.family.link.inverse(lp)

        return self.family.loglike(self.endog.ravel(), expval.ravel())


    def score(self, params):

        _, qm, resid, lp = self._linpred(params)

        # The fitted means
        mu = self.family.fitted(lp)

        # The derivative of the log-likelihood with respect to
        # the canonical parameters
        sf = (self.endog - mu) / self.family.link.deriv(mu)
        sf /= self.family.variance(mu)

        # The score with respect to the intercept
        si = sf.sum(0)
        si -= np.dot(np.dot(sf, qm), qm.T).sum(0)

        # The score with respect to the factors
        rts = np.dot(resid.T, sf)
        df = np.dot(rts, qm) + np.dot(rts.T, qm)

        sc = np.concatenate((si, df.ravel()))

        return sc


    def fit(self, maxiter=1000, gtol=1e-8):

        n, p = self.endog.shape
        d = self.ndim

        # Starting values
        icept = self.satparam.mean(0)
        cnp = self.satparam - icept
        _, _, vt = np.linalg.svd(cnp,0)
        v = vt.T
        v = v[:, 0:d]
        pa = np.concatenate((icept, v.ravel()))

        converged = False

        for itr in range(maxiter):

            icept = pa[0:p]
            qm = pa[p:].reshape((p, d))
            grad = self.score(pa)

            if itr % 2 == 0:

                # If we fail to update both the intecept
                # and the factor matrix, then stop
                fail1, fail2 = False, False

                # Update the intercept parameters
                gm = grad[0:p]
                gsn = np.sum(gm**2)

                def f(t):
                    qp = pa.copy()
                    qp[0:p] += t*gm
                    return qp, self.loglike(qp)

                # Take an uphill step
                t = 1.0
                _, f0 = f(0)
                while t > 1e-14:
                    qp, f1 = f(t)
                    if f1 > f0:
                        pa = qp
                        break
                    t /= 2

                if t <= 1e-14:
                    fail1 = True
                continue

            # Update the projection parameters
            gf = grad[p:].reshape((p, d))

            gf -= np.dot(qm, np.dot(qm.T, gf))
            gsn += np.sum(gf**2)
            u, s, vt = np.linalg.svd(gf, 0)
            v = vt.T

            def f(t):
                co = np.cos(s*t)
                si = np.sin(s*t)
                qq = np.dot(qm, np.dot(v, co[:, None] * vt)) + np.dot(u, si[:, None] * vt)
                qp = np.concatenate((icept, qq.ravel()))
                return qp, self.loglike(qp)

            # Take an uphill step
            t = 1.0
            _, f0 = f(0)
            while t > 1e-14:
                qp, f1 = f(t)
                if f1 > f0:
                    pa = qp
                    break
                t /= 2

            if t <= 1e-14:
                fail2 = True

            if np.sqrt(gsn) < gtol:
                converged = True
                break

            if fail1 and fail2:
                break

        if not converged:
            warnings.warn("GPCA did not converge")

        icept = pa[0:p]
        qm = pa[p:].reshape((p, d))

        results = GPCAResults(icept, qm)
        results.converged = converged
        results.params = pa
        results.score_norm = np.sqrt(gsn)

        return results


class GPCAResults:

    def __init__(self, intercept, factors):

        self.intercept = intercept
        self.factors = factors


def test_score():

    from numpy.testing import assert_allclose

    np.random.seed(23424)
    n = 100
    p = 5

    for j in 0, 1:
        for d in 1, 2, 3:
            for k in range(10):

                if j == 0:
                    endog = np.random.normal(size=(n, p))
                    pca = GPCA(endog, d)
                    mn = np.random.normal(size=p)
                else:
                    endog = np.random.poisson(100, size=(n, p))
                    pca = GPCA(endog, d, family=families.Poisson())
                    mn = np.log(100) + 0.5*np.random.normal(size=p)

                qm = np.random.normal(size=(p, d))
                qm, _, _ = np.linalg.svd(qm, 0)

                params = np.concatenate((mn, qm.ravel()))

                ll = pca.loglike(params)
                score = pca.score(params)

                # Numeric derivative
                nscore = np.zeros(p + p*d)
                f = 1e-7
                for j in range(p +p*d):
                    params1 = params.copy()
                    params1[j] += f
                    nscore[j] = (pca.loglike(params1) - pca.loglike(params)) / f

                assert_allclose(nscore, score, atol=1e-3, rtol=1e-4)

def test_fit_gaussian():

    from numpy.testing import assert_allclose

    np.random.seed(23424)
    n = 100
    p = 5

    for d in range(1, 6):
        endog = np.random.normal(size=(n, p)) + np.random.normal(size=p)
        pca = GPCA(endog, d)
        r = pca.fit(gtol=1e-12)
        icept, fac = r.intercept, r.factors

        resid0 = endog - icept
        fv = np.dot(resid0, np.dot(fac, fac.T)) + icept

        icept1 = endog.mean(0)
        endogc = endog - icept1
        u, s, vt = np.linalg.svd(endogc, 0)
        s[d:] = 0
        fv0 = np.dot(u, np.dot(np.diag(s), vt)) + icept1
        fac1 = vt.T[:, 0:d]

        p1 = np.dot(fac, fac.T)
        p2 = np.dot(fac1, fac1.T)

        assert(np.abs(np.trace(np.dot(p1, p2)) - d) < 1e-4)
        assert(np.max(np.abs(fv - fv0)) < 1e-10)
        assert(r.converged)


def test_fit_poisson():

    from numpy.testing import assert_allclose

    np.random.seed(23424)
    n = 1000
    p = 5

    for d in range(1, 6):
        icept = np.linspace(3, 5, p)
        fac = np.random.normal(size=(p, d))
        fac, _, _ = np.linalg.svd(fac, 0)
        sc = np.random.normal(size=(n, d))
        lp = np.dot(sc, fac.T) + icept
        mu = np.exp(lp)

        endog = np.random.poisson(mu, size=(n, p))
        pca = GPCA(endog, d, family=families.Poisson())
        r = pca.fit()
        icept1, fac1 = r.intercept, r.factors

        p1 = np.dot(fac, fac.T)
        p2 = np.dot(fac1, fac1.T)

        assert(np.abs(d - np.trace(np.dot(p1, p2))) < 1e-2)
        assert(r.score_norm < 0.005)


def test():
    test_score()
    test_fit_gaussian()
    test_fit_poisson()
