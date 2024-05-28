import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import tempfile
from collections import defaultdict
from scipy.integrate import quad

from jang.io import GW, NuDetector, Parameters
from jang.io.neutrinos import BackgroundGaussian, EffectiveAreaBase
import jang.utils.conversions
import jang.analysis.limits as limits
import jang.analysis.mcmc as mcmc
import jang.utils.flux as flux


tmpdir = tempfile.mkdtemp()

config_str = """
analysis:
  nside: 8
  apply_det_systematics: 0
  ntoys_det_systematics: 0
  search_region: region_90_excludezero
  likelihood: poisson
  prior_signal: flat

range:
  log10_flux: [-5, 5, 1000]
  log10_etot: [48, 62, 1400]
  log10_fnu: [-5, 10, 1500]
  neutrino_energy_GeV: [1, 1e6]
"""
config_file = f"{tmpdir}/config.yaml"
with open(config_file, "w") as f:
    f.write(config_str)
    
det_str = """
name: TestDet

nsamples: 1
samples:
  names: ["sampleA"]
  shortnames: ["A"]
  energyrange: [1, 1e6]

earth_location:
  latitude: 10.0
  longitude: 50.0
  units: deg

errors:
  acceptance: 0.00
  acceptance_corr: 1
"""
det_file = f"{tmpdir}/detector.yaml"
with open(det_file, "w") as f:
    f.write(det_str)

parameters = Parameters(config_file)
parameters.set_models("x**-2", jang.utils.conversions.JetIsotropic())
parameters.nside = 8
gw = GW(
        "GW190412", 
        "examples/input_files/gw_catalogs/GW190412/GW190412_PublicationSamples.fits", 
        "examples/input_files/gw_catalogs/GW190412/GW190412_subset.h5"
)
gw.set_parameters(parameters)

det = NuDetector(det_file)

class EffAreaTest(EffectiveAreaBase):
    acceptance = defaultdict(dict)
    nside = 8
    
    def evaluate(self, energy):
        return energy**2 * np.exp(-energy/10000)
    
    def integrate(self, spectrum):
        f_spectrum = eval("lambda x: %s" % spectrum)
        def func(x: float, *args):
            return self.evaluate(np.exp(x), *args) * f_spectrum(np.exp(x)) * np.exp(x)
        return quad(func, *self.sample.log_energy_range, limit=500)[0]
    
    def get_acceptance(self, ipix, spectrum):
        if ipix not in self.acceptance[spectrum]:
            self.acceptance[spectrum][ipix] = self.integrate(spectrum)
        return self.acceptance[spectrum][ipix]
        
det.samples[0].effective_area = EffAreaTest(det.samples[0])
det.samples[0].effective_area.nside = parameters.nside
acc = det.samples[0].effective_area.to_acceptance(det, parameters.nside, 0, parameters.spectrum)
det.set_acceptances([acc], "x**-2", nside=parameters.nside)
parameters.fspectrum = flux.FixedPowerLaw(1, 1000000, 2)

x = np.arange(1)
limits_old, limits_mcmc = [], []

for X in x:
    det.set_observations([X], [BackgroundGaussian(0.5, 0.1)])
    limits_old.append(limits.get_limit_flux(det, gw, parameters))
    limits.get_limit_etot(det, gw, parameters)
    limits_mcmc.append(mcmc.get_limits(det, gw, parameters)[0])
    
plt.close("all")
plt.plot(x, limits_old, label=r"Old", color="blue")
plt.plot(x, limits_mcmc, label=r"MCMC", color="orange")
plt.xlabel(r"Number of observed events")
plt.ylabel(r"$90\%$ upper limit")
plt.savefig("test.pdf")