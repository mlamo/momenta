import emcee
import numpy as np
from scipy.stats import multivariate_normal, poisson

from jang.io import NuDetectorBase, Transient, Parameters


class Model:
    
    def __init__(self, detector: NuDetectorBase, src: Transient, parameters: Parameters):
        self.nobs = np.array([s.nobserved for s in detector.samples])
        self.bkg = np.array([s.background for s in detector.samples])
        self.nsamples = detector.nsamples
        self.bkg_variations = parameters.apply_det_systematics
        self.acc_variations = parameters.apply_det_systematics and np.any(detector.error_acceptance != 0)
        if self.acc_variations:
            self.cov_acc = detector.error_acceptance
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
    
    def get_starting_points(self, n):
        p0 = np.random.rand(n, self.ndims)
        # flux normalizations
        accmax = np.array([np.max([s.effective_area.get_acceptance_map(c, self.nside) for s in self.detector.samples]) for c in self.flux.components])
        guesses = 10 / accmax
        i = 0
        p0[:,i:i+self.flux.ncomponents] *= guesses
        # GW random starting point
        i += self.flux.ncomponents
        p0[:,i] *= self.ntoys_src
        # Background scale
        if self.bkg_variations:
            i += 1
            p0[:,i:i+self.nsamples] = np.array([b.prior_transformed(p0[:,i+j]) for j,b in enumerate(self.bkg)]).T
        # Acceptance scale
        if self.acc_variations:
            i += self.nsamples
            p0[:,i:i+self.nsamples] = 1 + 1e-5 * p0[:,i:i+self.nsamples]
        return p0
        
    def log_prob(self, x):

        i = 0
        norm = x[i:i+self.flux.ncomponents]
        i += self.flux.ncomponents
        itoy = int(np.floor(x[i]))
        i += 1
        if self.bkg_variations:
            nbkg = x[i:i+self.nsamples]
            i += self.nsamples
        else:
            nbkg = np.array([b.nominal for b in self.bkg])
        if self.acc_variations:
            facc = x[i:i+self.nsamples]
        else:
            facc = 1
            
        # prior boundaries
        for n in norm:
            if n < 0:
                return -np.inf
        if not (0 <= itoy < self.ntoys_src):
            return -np.inf
        if np.any(nbkg < 0) or np.any(facc < 0):
            return -np.inf

        # likelihood
        toy = self.toys_src.iloc[itoy]
        acc = [[s.effective_area.get_acceptance(c, int(toy["ipix"]), self.nside) / 6 for s in self.detector.samples] for c in self.flux.components]
        
        nsig = facc * np.array(norm).dot(acc)
        loglkl = np.sum(poisson.logpmf(self.nobs, nbkg + nsig))
        if self.pointsource:
            for i, s in enumerate(self.detector.samples):
                if s.events is None:
                    continue
                for ev in s.events:
                    loglkl += np.log(s.compute_event_probability(nsig[i], nbkg[i], ev, toy["ra"], toy["dec"]))
        if self.bkg_variations:
            loglkl += np.sum([b.logpdf(nbkg[i]) for i, b in enumerate(self.bkg)])
        if self.acc_variations:
            loglkl += multivariate_normal.logpdf(facc, mean=np.ones(self.nsamples), cov=self.cov_acc)
        return loglkl
    

def prepare_model(detector: NuDetectorBase, src: Transient, parameters: Parameters):
    return Model(detector, src, parameters)


def run_mcmc(model):

    nwalkers = 32
    sampler = emcee.EnsembleSampler(nwalkers, model.ndims, model.log_prob)
    state = sampler.run_mcmc(model.get_starting_points(nwalkers), 1000)
    sampler.reset()
    sampler.run_mcmc(state, 1000)
    samples = sampler.get_chain(flat=True)
    
    return {k: v for k, v in zip(model.param_names, samples.transpose())}