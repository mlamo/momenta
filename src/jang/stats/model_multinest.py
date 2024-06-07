from pymultinest.solve import solve
import numpy as np
from scipy.stats import poisson, norm

from jang.io import NuDetectorBase, Transient, Parameters


class Model:
    
    def __init__(self, detector: NuDetectorBase, src: Transient, parameters: Parameters):
        self.nobs = np.array([s.nobserved for s in detector.samples])
        self.bkg = np.array([s.background for s in detector.samples])
        self.nsamples = detector.nsamples
        self.bkg_variations = parameters.apply_det_systematics
        self.acc_variations = parameters.apply_det_systematics and np.any(detector.error_acceptance != 0)
        if self.acc_variations:
            self.chol_cov_acc = np.linalg.cholesky(detector.error_acceptance + 1e-5 * np.identity(self.nsamples))
        self.pointsource = (parameters.likelihood_method == "pointsource")
        self.jetmodel = parameters.jet
        self.flux = parameters.flux
        self.nside = parameters.nside
        self.toys_src = src.prepare_prior_samples(parameters.nside)
        self.ntoys_src = len(self.toys_src)
        self.detector= detector
        
    @property
    def ndims(self):
        nd = self.flux.ncomponents + 1  # flux + GW toy
        if self.bkg_variations:
            nd += self.nsamples  # background
        if self.acc_variations:
            nd += self.nsamples  # acceptance
        return nd
    
    @property
    def param_names(self):
        params = [f"phi{i}" for i in range(self.flux.ncomponents)]
        params += ["itoy"]
        if self.bkg_variations:
            params += [f"bkg{i}" for i in range(self.nsamples)]  # background
        if self.acc_variations:
            params += [f"facc{i}" for i in range(self.nsamples)]  # acceptance
        return params
    
    def prior(self, cube):
        x = cube.copy()
        i = 0
        x[i:i+self.flux.ncomponents] *= 100000
        i += self.flux.ncomponents
        x[i] = np.floor(self.ntoys_src * x[i])
        i += 1
        if self.bkg_variations:
            for j in range(self.nsamples):
                x[i+j] = self.bkg[j].prior_transformed(x[i+j])
            i += self.nsamples
        if self.acc_variations:
            rvs = norm.ppf(x[i:i+self.nsamples])
            x[i:i+self.nsamples] = np.identity(self.nsamples) + np.dot(self.chol_cov_acc, rvs)
        return x
        
    def loglike(self, cube):
        # Format input parameters
        i = 0
        norm = cube[i:i+self.flux.ncomponents]
        i += self.flux.ncomponents
        itoy = int(np.floor(cube[i]))
        i += 1
        if self.bkg_variations:
            nbkg = cube[i:i+self.nsamples]
            i += self.nsamples
        else:
            nbkg = [b.nominal for b in self.bkg]
        if self.acc_variations:
            facc = cube[i:i+self.nsamples]
        else:
            facc = 1
        # Get acceptance
        toy = self.toys_src.iloc[itoy]
        acc = [[s.effective_area.get_acceptance(c, int(toy["ipix"]), self.nside) for s in self.detector.samples] for c in self.flux.components]
        # Compute log-likelihood
        nsig = facc * np.array(norm).dot(acc) / 6
        loglkl = np.sum(poisson.logpmf(self.nobs, nbkg + nsig))
        if self.pointsource:
            for i, s in enumerate(self.detector.samples):
                if s.events is None:
                    continue
                for ev in s.events:
                    loglkl += np.log(s.compute_event_probability(nsig[i], nbkg[i], ev, toy["ra"], toy["dec"]))
        return loglkl


def prepare_model(detector: NuDetectorBase, src: Transient, parameters: Parameters):
    return Model(detector, src, parameters)


def run_mcmc(model):
    result = solve(LogLikelihood=model.loglike, Prior=model.prior, n_dims=model.ndims, verbose=False, sampling_efficiency=0.1)
    return {k: v for k, v in zip(model.param_names, result["samples"].transpose())}