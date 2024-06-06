from pymultinest.solve import solve
import numpy as np
from scipy.stats import poisson, norm

from jang.io import NuDetectorBase, GW, Parameters


class Model:
    
    def __init__(self, detector: NuDetectorBase, gw: GW, parameters: Parameters):
        self.nobs = np.array([s.nobserved for s in detector.samples])
        self.bkg = np.array([s.background for s in detector.samples])
        self.nsamples = detector.nsamples
        self.bkg_variations = parameters.apply_det_systematics
        self.acc_variations = parameters.apply_det_systematics and np.any(detector.error_acceptance != 0)
        if self.acc_variations:
            self.chol_cov_acc = np.linalg.cholesky(detector.error_acceptance + 1e-5 * np.identity(self.nsamples))
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
    
    def prior(self, cube):
        i = 0
        cube[i:i+self.flux.ncomponents] *= 100000
        i += self.flux.ncomponents
        cube[i] = np.floor(self.ntoysgw * cube[i])
        i += 1
        if self.bkg_variations:
            for j in range(self.nsamples):
                cube[i+j] = self.bkg[j].prior_transformed(cube[i+j])
            i += self.nsamples
        if self.acc_variations:
            rvs = norm.ppf(cube[i:i+self.nsamples])
            cube[i:i+self.nsamples] = np.identity(self.nsamples) + np.dot(self.chol_cov_acc, rvs)
        return cube
        
    def loglike(self, cube):
        i = 0
        norm = cube[i:i+self.flux.ncomponents]
        i += self.flux.ncomponents
        itoygw = int(np.floor(cube[i]))
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
        ipix = int(self.toysgw[itoygw][0])
        acc = self.accs[:,:,ipix] / 6
        return np.sum(poisson.logpmf(self.nobs, nbkg + facc * np.array(norm).dot(acc)))


def prepare_model(detector: NuDetectorBase, gw: GW, parameters: Parameters):
    return Model(detector, gw, parameters)


def run_mcmc(model):
    result = solve(LogLikelihood=model.loglike, Prior=model.prior, n_dims=model.ndims, verbose=False, sampling_efficiency=0.1)    
    return {"phi0": result["samples"].transpose()[0]}