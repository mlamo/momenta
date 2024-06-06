import emcee
import healpy as hp
import numpy as np
from scipy.stats import multivariate_normal, poisson


from jang.io import NuDetectorBase, GW, Parameters
from jang.io.neutrinos import BackgroundFixed, BackgroundGaussian


class Model:
    
    def __init__(self, detector: NuDetectorBase, gw: GW, parameters: Parameters):
        self.nobs = np.array([s.nobserved for s in detector.samples])
        self.bkg = np.array([s.background for s in detector.samples])
        self.nsamples = detector.nsamples
        self.bkg_variations = parameters.apply_det_systematics
        self.acc_variations = parameters.apply_det_systematics and np.any(detector.error_acceptance != 0)
        if self.acc_variations:
            self.cov_acc = detector.error_acceptance
        self.flux = parameters.flux
        self.toysgw = gw.prepare_toys("ra", "dec", "luminosity_distance", "radiated_energy", "theta_jn", nside=parameters.nside)
        self.toysgw = np.array([[toy.ipix, toy.luminosity_distance, toy.radiated_energy, toy.theta_jn] for toy in self.toysgw])
        self.ntoysgw = len(self.toysgw)
        self.accs = np.array([detector.get_acceptance_maps(c, parameters.nside) for c in self.flux.components])
        
    @property
    def ndims(self):
        nd = self.flux.ncomponents + 1  # flux + GW toy
        if self.bkg_variations:
            nd += self.nsamples  # background
        if self.acc_variations:
            nd += self.nsamples  # acceptance
        return nd
    
    def get_starting_points(self, n):
        p0 = np.random.rand(n, self.ndims)
        # flux normalizations
        guesses = 10 * 1/np.max(np.average(self.accs, axis=2), axis=1)
        i = 0
        p0[:,i:i+self.flux.ncomponents] *= guesses
        # GW random starting point
        i += self.flux.ncomponents
        p0[:,i] *= self.ntoysgw
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
        itoygw = int(np.floor(x[i]))
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
        if not (0 <= itoygw < self.ntoysgw):
            return -np.inf
        if np.any(nbkg < 0) or np.any(facc < 0):
            return -np.inf

        # likelihood
        ipix = int(self.toysgw[itoygw][0])
        acc = self.accs[:,:,ipix] / 6
                
        loglkl = np.sum(poisson.logpmf(self.nobs, nbkg + facc * np.array(norm).dot(acc)))
        if self.bkg_variations:
            loglkl += np.sum([b.logpdf(nbkg[i]) for i, b in enumerate(self.bkg)])
        if self.acc_variations:
            loglkl += multivariate_normal.logpdf(facc, mean=np.ones(self.nsamples), cov=self.cov_acc)
        return loglkl


def prepare_model(detector: NuDetectorBase, gw: GW, parameters: Parameters):
    return Model(detector, gw, parameters)


def run_mcmc(model):

    nwalkers = 32
    sampler = emcee.EnsembleSampler(nwalkers, model.ndims, model.log_prob)
    state = sampler.run_mcmc(model.get_starting_points(nwalkers), 10000)
    sampler.reset()
    sampler.run_mcmc(state, 10000)
    samples = sampler.get_chain(flat=True)
    
    # import matplotlib.pyplot as plt
    # plt.close("all")
    # _, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
    # for i in range(nwalkers):
    #     for j in range(2):
    #         ax[j].plot(sampler.get_chain()[:,i,j])
    # ax[0].set_xlabel(r"Step #")
    # ax[1].set_xlabel(r"Step #")
    # ax[0].set_ylabel(r"Norm[0]")
    # ax[1].set_ylabel(r"itoygw")
    # ax[0].set_yscale("log")
    # plt.tight_layout()
    # plt.savefig("test.png")
    
    return {"phi0": samples[:,0]}