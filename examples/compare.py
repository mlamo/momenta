import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import time

from collections import defaultdict

from momenta.io import GW, NuDetector, Parameters
from momenta.io.neutrinos import BackgroundGaussian, BackgroundPoisson, EffectiveAreaAllSky
from momenta.stats.run import run_ultranest
from momenta.stats.limits import get_limits, get_limits_with_uncertainties
from momenta.stats.bayes_factor import compute_log_bayes_factor_tobkg
import momenta.utils.conversions
import momenta.utils.flux as flux

matplotlib.use("Agg")


tmpdir = tempfile.mkdtemp()


def test_onesample(src, parameters):
    
    det_str = """
    name: TestDet
    samples: ["A"]
    errors:
        acceptance: 0.10
        acceptance_corr: 0
    """
    det_file = f"{tmpdir}/detector.yaml"
    with open(det_file, "w") as f:
        f.write(det_str)
    
    det = NuDetector(det_file)

    class EffAreaTest1(EffectiveAreaAllSky):
        def evaluate(self, energy, ipix, nside):
            return energy**2 * np.exp(-energy/10000)

    det.set_effective_areas([EffAreaTest1()])
    det.set_observations([0], [BackgroundGaussian(0.5, 0.1)])
    
    results = defaultdict(list)
    uncertainties = defaultdict(list)
    times = defaultdict(lambda: 0)
    
    N = 20
    parameters.apply_det_systematics = False
    parameters.prior_normalisation = "flat-linear"
    for _ in range(N):
        t0 = time.time()
        model, result = run_ultranest(det, src, parameters)
        if "weighted_samples" in result:
            limits = get_limits_with_uncertainties(result["weighted_samples"], model)["flux0_norm"]
            limit, unc = limits[0], limits[1]
        else:
            limit = get_limits(result["samples"], model)["flux0_norm"]
            unc = np.nan
        results["flatlin"].append(limit)
        uncertainties["flatlin"].append(unc)
        times["flatlin"] += time.time() - t0

    N = 20
    parameters.apply_det_systematics = False
    parameters.prior_normalisation = "flat-log"
    for _ in range(N):
        t0 = time.time()
        model, result = run_ultranest(det, src, parameters)
        if "weighted_samples" in result:
            limits = get_limits_with_uncertainties(result["weighted_samples"], model)["flux0_norm"]
            limit, unc = limits[0], limits[1]
        else:
            limit = get_limits(result["samples"], model)["flux0_norm"]
            unc = np.nan
        results["flatlog"].append(limit)
        uncertainties["flatlog"].append(unc)
        times["flatlog"] += time.time() - t0
            
    N = 20
    parameters.apply_det_systematics = False
    parameters.prior_normalisation = "jeffreys-pois"
    for _ in range(N):
        t0 = time.time()
        model, result = run_ultranest(det, src, parameters)
        if "weighted_samples" in result:
            limits = get_limits_with_uncertainties(result["weighted_samples"], model)["flux0_norm"]
            limit, unc = limits[0], limits[1]
        else:
            limit = get_limits(result["samples"], model)["flux0_norm"]
            unc = np.nan
        results["jeffreyspois"].append(limit)
        uncertainties["jeffreyspois"].append(unc)
        times["jeffreyspois"] += time.time() - t0
            
    # compute naive upper limits
    nside = 8
    best_ipix = np.argmax(gw.fits.get_skymap(nside))
    parameters.flux.set_shapes([2])
    acc = EffAreaTest1().compute_acceptance(parameters.flux.components[0], best_ipix, nside)
    print(f"Naive UL = {2.3 / (acc/6):.2e}")

    for k in results.keys():
        print(f"{k:25s} => {np.average(results[k]):.2e} ± {np.std(results[k]):.2e} ({np.average(uncertainties[k]):.2e}), TIME = {times[k]/N:.2f} s")


def test_bayesfactor(src, parameters):
    
    det_str = """
    name: TestDet
    samples: ["A", "B"]
    errors:
        acceptance: 0.10
        acceptance_corr: 0
    """
    det_file = f"{tmpdir}/detector.yaml"
    with open(det_file, "w") as f:
        f.write(det_str)
    
    det = NuDetector(det_file)

    class EffAreaTest1(EffectiveAreaAllSky):
        def evaluate(self, energy, ipix, nside):
            return energy**2 * np.exp(-energy/10000)

    det.set_effective_areas([EffAreaTest1(), EffAreaTest1()])
    det.set_observations([0, 0], [BackgroundPoisson(20, 10), BackgroundPoisson(20, 10)])
    
    N1 = np.arange(0, 10+1, 3)
    N2 = np.arange(0, 10+1, 3)
    N1, N2 = np.meshgrid(N1, N2, indexing="ij")
    bf = np.zeros(N1.shape)
    
    for i in range(N1.shape[0]):
        for j in range(N2.shape[1]):
            det.samples[0].nobserved = N1[i,j]
            det.samples[1].nobserved = N2[i,j]
            _, result = run_ultranest(det, src, parameters)
            bf[i, j] = compute_log_bayes_factor_tobkg(result, det, src, parameters)
            print(N1[i,j], N2[i,j], bf[i,j])
    
    np.savetxt("test.csv", bf)
    
    plt.pcolormesh(N1, N2, bf, shading="nearest", cmap="Greens")
    plt.xlabel("Number of observed events in sample 1")
    plt.ylabel("Number of observed events in sample 2")
    plt.savefig("test.png")


if __name__ == "__main__":
    
    config_str = """
    skymap_resolution: 8
    detector_systematics: 0

    analysis:
        likelihood: poisson
        prior_normalisation:
            type: flat-linear
            range: [1.0e-10, 1.0e+10]
    """
    config_file = f"{tmpdir}/config.yaml"
    with open(config_file, "w") as f:
        f.write(config_str)
    
    parameters = Parameters(config_file)
    parameters.set_models(flux.FluxFixedPowerLaw(1, 1e6, 2, eref=1), momenta.utils.conversions.JetVonMises(np.deg2rad(10)))
    gw = GW(
            name="GW190412", 
            path_to_fits="examples/input_files/gw_catalogs/GW190412/GW190412_PublicationSamples.fits", 
            path_to_samples="examples/input_files/gw_catalogs/GW190412/GW190412_subset.h5"
    )
    gw.set_parameters(parameters)
    
    test_onesample(gw, parameters)
    # test_bayesfactor(gw, parameters)