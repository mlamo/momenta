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
        self.detector = detector
        self.parameters = parameters
        self.flux = parameters.flux
        self.toys_src = src.prepare_prior_samples(parameters.nside)
        self.ntoys_src = len(self.toys_src)

    @property
    def ndims(self):
        nd = self.flux.nparameters + 1  # flux + GW toy
        if self.bkg_variations:
            nd += self.nsamples  # background
        if self.acc_variations:
            nd += self.nsamples  # acceptance
        return nd

    @property
    def param_names(self):
        params = [f"flux{i}_norm" for i in range(self.flux.ncomponents)]
        params += [f"flux{i}_{s}" for i, c in enumerate(self.flux.components) for s in c.shape_names]
        params += ["itoy"]
        if self.bkg_variations:
            params += [f"bkg{i}" for i in range(self.nsamples)]  # background
        if self.acc_variations:
            params += [f"facc{i}" for i in range(self.nsamples)]  # acceptance
        return params
    
    def get_starting_points(self, n):
        p0 = np.random.rand(n, self.ndims)
        # flux shapes
        p0[:,self.flux.ncomponents:self.flux.nparameters] = [self.flux.prior_transform(pp0[self.flux.ncomponents:self.flux.nparameters]) for pp0 in p0]
        i = 0
        for j in range(n):
            self.flux.set_shapes(p0[j:,self.flux.ncomponents:self.flux.nparameters])
            accmax = np.array([np.max([s.effective_area.get_acceptance_map(c, self.parameters.nside) for s in self.detector.samples]) for c in self.flux.components])
            guesses = 10 / accmax
            p0[j,i:i+self.flux.ncomponents] *= guesses
        i += self.flux.nparameters
        # GW random starting point
        p0[:,i] *= self.ntoys_src
        # Background scale
        if self.bkg_variations:
            i += 1
            p0[:,i:i+self.nsamples] = np.array([b.prior_transform(p0[:,i+j]) for j,b in enumerate(self.bkg)]).T
        # Acceptance scale
        if self.acc_variations:
            i += self.nsamples
            p0[:,i:i+self.nsamples] = 1 + 1e-5 * p0[:,i:i+self.nsamples]
        return p0
        
    def log_prob(self, x):
        i = 0
        norm = x[i:i+self.flux.ncomponents]
        i += self.flux.ncomponents
        shapes = x[i:i+self.flux.nshapes]
        i += self.flux.nshapes
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
        # Prior boundaries
        for n in norm:
            if n < 0:
                return -np.inf
        for s, sb in zip(shapes, self.flux.shape_boundaries):
            if not (sb[0] <= s <= sb[1]):
                return -np.inf
        if not (0 <= itoy < self.ntoys_src):
            return -np.inf
        if np.any(nbkg < 0) or np.any(facc < 0):
            return -np.inf

        # Get acceptance
        self.flux.set_shapes(shapes)
        toy = self.toys_src.iloc[itoy]
        acc = [[s.effective_area.get_acceptance(c, int(toy["ipix"]), self.parameters.nside) / 6 for s in self.detector.samples] for c in self.flux.components]
        # Compute log-likelihood
        nsig = facc * np.array(norm).dot(acc)
        loglkl = np.sum(poisson.logpmf(self.nobs, nbkg + nsig))
        if self.parameters.likelihood_method == "pointsource":
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