import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import tempfile

from jang.io import GW, NuDetector, Parameters
from jang.io.neutrinos import BackgroundGaussian
import jang.utils.conversions
import jang.analysis.limits as limits
import jang.analysis.mcmc as mcmc


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
  neutrino_energy_GeV: [0.1, 1e8]
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
  energyrange: [0, 100]

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

acc = 6 * np.ones(hp.nside2npix(parameters.nside))
det.set_acceptances([acc], "x**-2", nside=parameters.nside)

x = np.arange(10+1)
limits_old, limits_mcmc = [], []

for X in x:
    det.set_observations([X], [BackgroundGaussian(0.5, 0.1)])
    limits_old.append(limits.get_limit_flux(det, gw, parameters))
    limits_mcmc.append(mcmc.get_limits(det, gw, parameters)[0])
    
plt.close("all")
plt.plot(x, limits_old, label=r"Old", color="blue")
plt.plot(x, limits_mcmc, label=r"MCMC", color="orange")
plt.xlabel(r"Number of observed events")
plt.ylabel(r"$90\%$ upper limit")
plt.savefig("test.pdf")