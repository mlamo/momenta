"""
    Copyright (C) 2024  Mathieu Lamoureux

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
from scipy.stats import norm, poisson

from momenta.io import NuDetectorBase, Transient, Parameters
from momenta.utils.conversions import solarmass_to_erg


def calculate_deterministics(samples, model):
    """Calculate different deterministic quantities:
    - eiso: total energy emitted in neutrinos assuming isotropic emission [in erg]
    - etot: total energy emitted in neutrinos assuming model=parameters.jet and using `theta_jn` as jet orientation w.r.t. Earth [in erg]
    - fnu: ratio between total energy in neutrinos `etot` and radiated energy in GW using `radiated_energy` [no units]
    """
    det = {}
    # Stop here if distance is not provided
    if "distance_scaling" not in model.toys_src.dtype.names:
        return det
    itoys = samples["itoy"].astype(int)
    nsamples = len(itoys)
    distance_scaling = model.toys_src[itoys]["distance_scaling"]
    norms = np.array([samples[f"flux{i}_norm"] for i in range(model.flux.ncomponents)])
    if model.flux.nshapevars > 0:
        shapes = np.array([samples[f"flux{i}_{s}"] for i, c in enumerate(model.flux.components) for s in c.shape_names])
        det["eiso"] = np.empty(nsamples)
        for isample in range(nsamples):
            model.flux.set_shapevars(shapes[:, isample])
            det["eiso"][isample] = np.sum(norms * model.flux.flux_to_eiso(distance_scaling[isample]))
    else:
        det["eiso"] = np.sum(norms + model.flux.flux_to_eiso(distance_scaling), axis=0)
    if "radiated_energy" in model.toys_src.dtype.names:
        radiated_energy = model.toys_src[itoys]["radiated_energy"]
        det["fnuiso"] = det["eiso"] / (radiated_energy * solarmass_to_erg)
    # Stop here if jet model is not provided or if `theta_jn`` is missing
    if model.parameters.jet is None or "theta_jn" not in model.toys_src.dtype.names:
        return det
    theta_jn = model.toys_src[itoys]["theta_jn"]
    det["etot"] = det["eiso"] / model.parameters.jet.etot_to_eiso(theta_jn)
    if "radiated_energy" in model.toys_src.dtype.names:
        radiated_energy = model.toys_src[itoys]["radiated_energy"]
        det["fnu"] = det["etot"] / (radiated_energy * solarmass_to_erg)
    return det


class ModelNested:

    def __init__(self, detector: NuDetectorBase, src: Transient, parameters: Parameters):
        self.nobs = np.array([s.nobserved for s in detector.samples])
        self.bkg = np.array([s.background for s in detector.samples])
        self.nsamples = detector.nsamples
        self.bkg_variations = parameters.apply_det_systematics
        self.acc_variations = parameters.apply_det_systematics and np.any(detector.error_acceptance != 0)
        if self.acc_variations:
            self.chol_cov_acc = np.linalg.cholesky(detector.error_acceptance + 1e-5 * np.identity(self.nsamples))
        self.detector = detector
        self.parameters = parameters
        self.flux = parameters.flux
        self.toys_src = src.prepare_prior_samples(parameters.nside)
        self.ntoys_src = len(self.toys_src)
        self.fluxnorm_prior = self.parameters.prior_normalisation
        self.fluxnorm_range = self.parameters.prior_normalisation_range

    @property
    def ndims(self):
        nd = self.flux.nparameters + 1  # flux (norms + shapes) + GW toy
        if self.bkg_variations:
            nd += self.nsamples  # background
        if self.acc_variations:
            nd += self.nsamples  # acceptance
        return nd

    @property
    def param_names(self):
        params = [f"flux{i}_norm" for i in range(self.flux.ncomponents)]
        params += [f"flux{i}_{s}" for i, c in enumerate(self.flux.components) for s in c.shapevar_names]
        params += ["itoy"]
        if self.bkg_variations:
            params += [f"bkg{i}" for i in range(self.nsamples)]  # background
        if self.acc_variations:
            params += [f"facc{i}" for i in range(self.nsamples)]  # acceptance
        return params

    def prior_norm(self, cube):
        if self.fluxnorm_prior == "flat-linear":
            return self.fluxnorm_range[0] + (self.fluxnorm_range[1] - self.fluxnorm_range[0]) * cube
        elif self.fluxnorm_prior == "flat-log":
            return np.power(10, np.log10(self.fluxnorm_range[0]) + (np.log10(self.fluxnorm_range[1]) - np.log10(self.fluxnorm_range[0])) * cube)
        elif self.fluxnorm_prior == "jeffreys":
            return self.fluxnorm_range[0] + (self.fluxnorm_range[1] - self.fluxnorm_range[0]) * cube

    def prior(self, cube):
        """Convert from unit hypercube to hyperparameter space following the prior distributions.
        
        Args:
            cube (np.ndarray): unit cube of dimension = (N, D) where N is the number of points to evaluate and D the number of dimensions
        
        Returns:
            np.ndarray: same dimension as input, but values in real parameter space
        """
        x = cube.copy()
        i = 0
        x[..., i : i + self.flux.ncomponents] = self.prior_norm(x[..., i : i + self.flux.ncomponents])
        i += self.flux.ncomponents
        x[..., i : i + self.flux.nshapevars] = self.flux.prior_transform(x[..., i : i + self.flux.nshapevars])
        i += self.flux.nshapevars
        x[..., i] = np.floor(self.ntoys_src * x[..., i])
        i += 1
        if self.bkg_variations:
            for j in range(self.nsamples):
                x[..., i + j] = self.bkg[j].prior_transform(x[..., i + j])
            i += self.nsamples
        if self.acc_variations:
            rvs = norm.ppf(x[..., i : i + self.nsamples])
            x[..., i : i + self.nsamples] = np.ones(self.nsamples) + np.dot(rvs, self.chol_cov_acc)
        return x

    def loglike(self, cube):
        """Compute the log-likelihood.
        
        Args:
            cube (np.ndarray): parameter hypercube dimension = (N, D) where N is the number of points to evaluate and D the number of dimensions
            
        Returns:
            np.ndarray: value of log-likelihood for the N points
        """
        npoints = cube.shape[0]
        # INPUTS
        # > flux normalisation parameters
        i = 0
        norms = cube[:, i : i + self.flux.ncomponents]  # dims = (npoints, ncompflux)
        i += self.flux.ncomponents
        # > flux shape parameters
        shapes = cube[:, i : i + self.flux.nshapevars]  # dims = (npoints, nshapes)
        i += self.flux.nshapevars
        # > source parameter
        itoys = np.floor(cube[:, i]).astype(int)
        toys = self.toys_src[itoys]
        i += 1
        # > background parameters
        if self.bkg_variations:
            nbkg = cube[:, i : i + self.nsamples]  # dims = (npoints, nsamples)
            i += self.nsamples
        else:
            nbkg = np.tile([b.nominal for b in self.bkg], (npoints, 1))
        # > acceptance variation parameters
        if self.acc_variations:
            facc = cube[:, i : i + self.nsamples]  # dims = (npoints, nsamples)
        else:
            facc = np.ones((npoints, self.nsamples))
        # ACCEPTANCE
        accs = np.zeros((npoints, self.flux.ncomponents, self.detector.nsamples))  # dims = (npoints, ncompflux, nsamples)
        for ipoint in range(npoints):
            ishape = 0
            for iflux, c in enumerate(self.flux.components):
                if c.nshapevars > 0:
                    c.set_shapevars(shapes[ipoint, ishape : ishape + c.nshapevars])
                    ishape += c.nshapevars
                for isample, s in enumerate(self.detector.samples):
                    accs[ipoint, iflux, isample] = s.effective_area.get_acceptance(c, toys[ipoint].ipix, self.parameters.nside)
        # LOG-LIKELIHOOD
        nsigs = facc[:, np.newaxis, :] * (norms[:, :, np.newaxis] * accs / 6)  # dims = (npoints, ncompflux, nsamples)
        nexps = nbkg + np.sum(nsigs, axis=1)  # dims = (npoints, nsamples)
        if self.parameters.likelihood_method == "poisson":
            loglkl = np.sum(-nexps + self.nobs * np.log(nexps), axis=1)  # dims = (npoints, )
        if self.parameters.likelihood_method == "pointsource":
            loglkl = np.sum(-nexps, axis=1)  # dims = (npoints, )
            for isample, s in enumerate(self.detector.samples):
                if s.events is None:
                    loglkl += self.nobs[isample] * np.log(nexps[:, isample])  # dims = (npoints, )
                    continue
                psigs = np.zeros((npoints, self.flux.ncomponents, s.nobserved))  # dims = (npoints, ncompflux, nevents)
                ishape = 0
                for iflux, c in enumerate(self.flux.components):
                    for ipoint in range(npoints):
                        if c.nshapevars > 0:
                            c.set_shapevars(shapes[ipoint, ishape : ishape + c.nshapevars])
                        for ievt, evt in enumerate(s.events):
                            psigs[ipoint, iflux, ievt] = s.compute_signal_probability(evt, c, toys[ipoint].ra, toys[ipoint].dec)
                    ishape += c.nshapevars
                pbkgs = np.zeros(s.nobserved)
                for ievt, evt in enumerate(s.events):
                    pbkgs[ievt] = s.compute_background_probability(evt)
                probs = nbkg[:, isample, np.newaxis] * pbkgs + np.sum(nsigs[:, :, isample, np.newaxis] * psigs, axis=1)
                loglkl += np.sum(np.log(probs), axis=1)
        return loglkl


class ModelNested_BkgOnly:
    """Same model as `ModelNested` but only with the background (used for Bayes factor computation)."""

    def __init__(self, detector: NuDetectorBase, parameters: Parameters):
        self.nobs = np.array([s.nobserved for s in detector.samples])
        self.bkg = np.array([s.background for s in detector.samples])
        self.nsamples = detector.nsamples
        self.bkg_variations = parameters.apply_det_systematics
        self.detector = detector
        self.parameters = parameters

    @property
    def ndims(self):
        nd = 0
        if self.bkg_variations:
            nd += self.nsamples  # background
        return nd

    @property
    def param_names(self):
        params = []
        if self.bkg_variations:
            params += [f"bkg{i}" for i in range(self.nsamples)]  # background
        return params

    def prior(self, cube):
        x = cube.copy()
        if self.bkg_variations:
            for j in range(self.nsamples):
                x[j] = self.bkg[j].prior_transform(x[j])
        return x

    def loglike(self, cube):
        # Format input parameters
        if self.bkg_variations:
            nbkg = cube
        else:
            nbkg = [b.nominal for b in self.bkg]
        # Compute log-likelihood
        loglkl = np.sum(poisson.logpmf(self.nobs, nbkg))
        return loglkl
