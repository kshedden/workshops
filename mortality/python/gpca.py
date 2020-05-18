import numpy as np
import statsmodels
import statsmodels.base.model as base
from statsmodels.genmod import families
import warnings

class GPCA(base.LikelihoodModel):

    def __init__(self, endog, ndim, offset=None, family=None, penmat=None):
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

        if offset is not None:
            if offset.shape != endog.shape:
                msg = "endog and offset must have the same shape"
                raise ValueError(msg)
            self.offset = np.asarray(offset)

        if penmat is not None:
            pm = []
            if len(penmat) != 2:
                msg = "penmat must be a tuple of length 2"
                raise ValueError(msg)
            for j in range(2):
                if np.isscalar(penmat[j]):
                    n, p = endog.shape
                    pm.append(self._gen_penmat(penmat[j], n, p))
                else:
                    pm.append(penmat[j])
            self.penmat = pm

        # Calculate the saturated parameter
        if isinstance(family, families.Poisson):
            satparam = np.where(endog != 0, np.log(endog), -3)
        elif isinstance(family, families.Binomial):
            satparam = np.where(endog == 1, 3, -3)
        elif isinstance(family, families.Gaussian):
            satparam = endog
        else:
            raise ValueError("Unknown family")
        self.satparam = satparam


    def _gen_penmat(self, f, n, p):
        pm = np.zeros((p-2, p))
        for k in range(p-2):
            pm[k, k:k+3] = [-1, 2, -1]
        return f * pm * n

    def _linpred(self, params):

        n, p = self.endog.shape

        icept = params[0:p]
        qm = params[p:].reshape((p, self.ndim))

        resid = self.satparam - icept
        if hasattr(self, "offset"):
            resid -= self.offset

        lp = icept + np.dot(np.dot(resid, qm), qm.T)
        if hasattr(self, "offset"):
            lp += self.offset

        return icept, qm, resid, lp

    def _flip(self, params):
        """
        Multiply factors by -1 so that the majority of entries
        are positive.
        """
        p = self.endog.shape[1]
        icept = params[0:p]
        qm = params[p:].reshape((p, self.ndim))
        for j in range(self.ndim):
            if np.sum(qm[:, j] < 0) > np.sum(qm[:, j] > 0):
                qm[:, j] *= -1
        return np.concatenate((icept, qm.ravel()))

    def predict(self, params, linear=False):
        """
        Return the fitted mean or its linear predictor.

        Parameters
        ----------
        params : array-like
            The parameters to use to produce the fitted mean
        linear : boolean
            If true, return the linear predictor, otherwise
            return the fitted mean, which is the inverse
            link function evaluated at the linear predictor.

        Returns an array with the same shape as endog, containing
        fitted values corresponding to the given parameters.

        Notes
        -----
        If an offset is present, it is included in the linear
        predictor.
        """

        _, _, _, lp = self._linpred(params)

        if linear:
            return lp
        else:
            return self.family.fitted(lp)


    def scores(self, params):
        """
        Returns the PC scores for each case.

        Parameters
        ----------
        params : array-like
            The parameters at which the scores are
            calculated.

        Returns
        -------
        An array of scores.
        """

        _, qm, resid, _ = self._linpred(params)

        return np.dot(resid, qm)


    def loglike(self, params):

        icept, qm, _, lp = self._linpred(params)
        expval = self.family.link.inverse(lp)

        ll = self.family.loglike(self.endog.ravel(), expval.ravel())

        if hasattr(self, "penmat"):
            pm = self.penmat
            ll -= np.sum(np.dot(pm[0], icept)**2)
            for j in range(self.ndim):
                ll -= np.sum(np.dot(pm[1], qm[:, j])**2)

        return ll

    def _orthog(self, qm, v):
        for i in range(5):
            v -= np.dot(qm, np.dot(qm.T, v))
            if np.max(np.abs(np.dot(qm.T, v))) < 1e-10:
                break
        return v


    def score(self, params, project=False):

        icept, qm, resid, lp = self._linpred(params)

        # The fitted means
        mu = self.family.fitted(lp)

        # The derivative of the log-likelihood with respect to
        # the canonical parameters
        sf = (self.endog - mu) / self.family.link.deriv(mu)
        sf /= self.family.variance(mu)

        # The score with respect to the intercept
        si = sf.sum(0)
        si = self._orthog(qm, si)

        # The score with respect to the factors
        rts = np.dot(resid.T, sf)
        df = np.dot(rts, qm) + np.dot(rts.T, qm)

        if hasattr(self, "penmat"):
            pm = self.penmat
            si -= 2 * np.dot(pm[0].T, np.dot(pm[0], icept))
            for j in range(self.ndim):
                df[:, j] -= 2 * np.dot(pm[1].T, np.dot(pm[1], qm[:, j]))

        if project:
            df = self._orthog(qm, df)

        sc = np.concatenate((si, df.ravel()))

        return sc

    def _update_icept(self, pa, gm, p):

        def f(t):
            qp = pa.copy()
            qp[0:p] += t*gm
            return qp, self.loglike(qp)

        _, f0 = f(0)

        # Take an uphill step
        def search(t, s, pa):
            while t > 1e-18:
                qp, f1 = f(t*s)
                if f1 > f0:
                    pa = qp
                    return pa, False
                t /= 2
            return pa, True

        pa, fail = search(1.0, 1, pa)

        return pa, fail

    def _update_factors(self, pa, icept, qm, gf):

        fail = False
        u, s, vt = np.linalg.svd(gf, 0)
        v = vt.T

        def f(t):
            co = np.cos(s*t)
            si = np.sin(s*t)
            qq = np.dot(qm, np.dot(v, co[:, None] * vt)) + np.dot(u, si[:, None] * vt)
            qp = np.concatenate((icept, qq.ravel()))
            return qp, self.loglike(qp)

        _, f0 = f(0)

        # Take an uphill step
        def search(t, s, pa):
            while t > 1e-14:
                qp, f1 = f(s*t)
                if f1 > f0:
                    pa = qp
                    return pa, True
                t /= 2
            return pa, False

        pa, fail = search(1.0, 1, pa)

        return pa, fail

    def fit(self, maxiter=1000, gtol=1e-8):

        n, p = self.endog.shape
        d = self.ndim

        # Starting values
        icept = self.satparam.mean(0)
        if hasattr(self, "offset"):
            icept -= self.offset.mean(0)
        cnp = self.satparam - icept
        if hasattr(self, "offset"):
            cnp -= (self.offset - self.offset.mean(0))
        _, _, vt = np.linalg.svd(cnp,0)
        v = vt.T
        v = v[:, 0:d]
        pa = np.concatenate((icept, v.ravel()))

        converged = False

        for itr in range(maxiter):

            if itr % 2 == 0:

                # If we fail to update both the intecept
                # and the factor matrix, then stop
                fail1, fail2 = False, False

                for ii in range(3):
                    grad = self.score(pa)
                    gm = grad[0:p]
                    if np.sqrt(np.sum(gm**2)) < 1e-4:
                        break
                    pa, fail1 = self._update_icept(pa, gm, p)

                # Update the intercept parameters
                gm = grad[0:p]
                gsn = np.sum(gm**2)

                continue

            for ii in range(3):
                icept = pa[0:p]
                qm = pa[p:].reshape((p, d))
                grad = self.score(pa)

                # Update the projection parameters
                gf = grad[p:].reshape((p, d))

                # Orthogonalize twice due to roundoff
                gf -= np.dot(qm, np.dot(qm.T, gf))
                gf -= np.dot(qm, np.dot(qm.T, gf))
                if np.sqrt(np.sum(gf**2)) < 1e-4:
                    break

                pa, fail2 = self._update_factors(pa, icept, qm, gf)

            gsn += np.sum(gf**2)

            if np.sqrt(gsn) < gtol:
                converged = True
                break

            if fail1 and fail2:
                break

        if not converged:
            warnings.warn("GPCA did not converge")

        pa = self._flip(pa)
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
        endog += 10*np.arange(p)
        pca = GPCA(endog, d)
        r = pca.fit(gtol=1e-10)
        assert(r.converged)
        icept, fac = r.intercept, r.factors

        # GPCA fitted values
        resid0 = endog - icept
        fv = np.dot(resid0, np.dot(fac, fac.T)) + icept

        # PCA fitted values
        icept1 = endog.mean(0)
        endogc = endog - icept1
        u, s, vt = np.linalg.svd(endogc, 0)
        s[d:] = 0
        fv0 = np.dot(u, np.dot(np.diag(s), vt)) + icept1
        fac1 = vt.T[:, 0:d]

        # Check that PCA and GPCA factors agree
        p1 = np.dot(fac, fac.T)
        p2 = np.dot(fac1, fac1.T)
        assert_allclose(np.trace(np.dot(p1, p2)), d, rtol=1e-10,
                        atol=1e-10)

        # Check that PCA and GPCA fitted values agree
        assert_allclose(fv, fv0, rtol=1e-10, atol=1e-10)

        # The scores should be centered at zero
        scores = pca.scores(r.params)
        assert_allclose(scores.mean(0), 0, rtol=1e-8, atol=1e-8)

        # The GPCA scores and PCA scores should agree up to
        # ordering.
        scores1 = u[:, 0:d] * s[0:d]
        c = np.corrcoef(scores.T, scores1.T)
        assert_allclose(np.abs(c).sum(0), 2*np.ones(2*d))

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
        r = pca.fit(maxiter=50)
        icept1, fac1 = r.intercept, r.factors

        # Check intercepts
        assert_allclose(icept, icept1, atol=1e-2, rtol=1e-1)

        # Check factors
        p1 = np.dot(fac, fac.T)
        p2 = np.dot(fac1, fac1.T)
        assert_allclose(np.trace(np.dot(p1, p2)), d, atol=1e-2)

        # Scores should be approximately centered
        scores = pca.scores(r.params)
        assert_allclose(scores.mean(), 0, atol=1e-3)

        assert(r.score_norm < 0.005)


def test():
    test_score()
    test_fit_gaussian()
    test_fit_poisson()
