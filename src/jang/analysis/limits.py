"""Compute upper limits for any given Detector or SuperDetector (combination of Detector) and selected model, using the likelihoods."""

import copy
import healpy as hp
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Optional

from jang.io import GW, NuDetectorBase, Parameters
from jang.io.utils import get_search_region
import jang.stats
import jang.stats.posteriors

matplotlib.use("Agg")


def plot_posterior(x: np.ndarray, y: np.ndarray, limit: float, xlabel: str, outfile: str):

    plt.close("all")
    plt.plot(x, y)
    plt.xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel("Posterior")
    plt.axvline(limit, color="red", label=r"90% upper limit")
    plt.legend(loc="upper right")
    plt.savefig(outfile, dpi=300)


def get_limit_flux(detector: NuDetectorBase, gw: GW, parameters: Parameters, outfile: Optional[str] = None) -> float:
    """Return 90% upper limit on flux normalization (dn/dE = norm*{parameters.flux}).
    The related posterior will be saved in "{outfile}.[npy/png]" if provided."""

    variables = [jang.stats.PosteriorVariable("flux", *parameters.range_flux, log=True)]

    x, y = jang.stats.posteriors.compute_flux_posterior(variables, detector, gw, parameters)
    x, y = jang.stats.normalize(x[0], y)
    limit = jang.stats.compute_upperlimit_from_x_y(x, y)
    if outfile is not None:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        np.save(outfile + ".npy", [x, y])
        plot_posterior(x, y, limit, "Flux normalization", outfile + ".png")
    logging.getLogger("jang").info(
        "[Limits] %s, %s, %s, limit(Flux) = %.3e",
        gw.name,
        detector.name,
        parameters.flux.components[0],
        limit,
    )
    return limit


def get_limit_etot(detector: NuDetectorBase, gw: GW, parameters: Parameters, outfile: Optional[str] = None) -> float:
    """Return 90% upper limit on the total energy emitted in neutrinos (all-flavours) E(tot) [in erg].
    The related posterior will be saved in "{outfile}.[npy/png]" if provided."""

    variables = [jang.stats.PosteriorVariable("etot", *parameters.range_etot, log=True)]

    x, y = jang.stats.posteriors.compute_etot_posterior(variables, detector, gw, parameters)
    x, y = jang.stats.normalize(x[0], y)
    limit = jang.stats.compute_upperlimit_from_x_y(x, y)
    if outfile is not None:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        np.save(outfile + ".npy", [x, y])
        plot_posterior(x, y, limit, r"$E^{tot}_{\nu}$", outfile + ".png")
    logging.getLogger("jang").info(
        "[Limits] %s, %s, %s, %s, limit(Etot) = %.3e erg",
        gw.name,
        detector.name,
        parameters.jet.__repr__(),
        parameters.flux.components[0],
        limit,
    )
    return limit


def get_limit_fnu(detector: NuDetectorBase, gw: GW, parameters: Parameters, outfile: Optional[str] = None) -> float:
    """Return 90% upper limit on fnu=E(tot)/E(radiated).
    The related posterior will be saved in "{outfile}.[npy/png]" if provided."""

    variables = [jang.stats.PosteriorVariable("fnu", *parameters.range_fnu, log=True)]

    x, y = jang.stats.posteriors.compute_fnu_posterior(variables, detector, gw, parameters)
    x, y = jang.stats.normalize(x[0], y)
    limit = jang.stats.compute_upperlimit_from_x_y(x, y)
    if outfile is not None:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        np.save(outfile + ".npy", [x, y])
        plot_posterior(x, y, limit, r"$E^{tot}_{\nu}/E_{rad}$", outfile + ".png")
    logging.getLogger("jang").info(
        "[Limits] %s, %s, %s, %s, limit(fnu) = %.3e",
        gw.name,
        detector.name,
        parameters.jet.__repr__(),
        parameters.flux.components[0],
        limit,
    )
    return limit


def get_limitmap_flux(detector: NuDetectorBase, gw: GW, parameters: Parameters, outfile: Optional[str] = None, log: bool = False) -> float:
    """Return map 90% upper limit on flux normalization (dn/dE = norm*{parameters.flux}),
    without marginalizing over the source localization."""

    variables = [jang.stats.PosteriorVariable("flux", *parameters.range_flux, log=True)]

    region = get_search_region(detector, gw, parameters)
    limits = np.nan * np.ones(hp.nside2npix(parameters.nside))
    for ipix in region:
        x, y = jang.stats.posteriors.compute_flux_posterior(variables, detector, gw, parameters, fixed_gwpixel=ipix)
        x, y = jang.stats.normalize(x[0], y)
        limits[ipix] = jang.stats.compute_upperlimit_from_x_y(x, y)

    if outfile is not None:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        np.save(outfile + ".npy", limits)
        plt.close("all")
        print(np.nanmin(limits), np.nanmax(limits))
        int_min = np.floor(10 * float(f"{np.nanmin(limits):.3e}".split("e")[0]))
        power_min = int(f"{np.nanmin(limits):.3e}".split("e")[1])
        int_max = np.ceil(10 * float(f"{np.nanmax(limits):.3e}".split("e")[0]))
        power_max = int(f"{np.nanmax(limits):.3e}".split("e")[1])
        ax_min, ax_max = int_min * 10 ** (power_min - 1), int_max * 10 ** (power_max - 1)
        if log:
            hp.mollview(
                np.log10(limits),
                cmap="Blues",
                unit=r"$\log_{10}$(limit on flux [GeV/cm$^2$])",
                rot=180,
                title="",
                min=np.log10(ax_min),
                max=np.log10(ax_max),
                format="%.2f",
            )
        else:
            hp.mollview(limits, cmap="Blues", unit=r"Limit on flux [GeV/cm$^2$]", rot=180, title="", min=ax_min, max=ax_max)
        hp.graticule()
        plt.savefig(outfile + ".png", dpi=300)

    return limits
