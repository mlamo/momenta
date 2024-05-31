"""Formatting the results in a Pandas database and related plotting functions."""

import copy
import logging
from math import floor, ceil
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from typing import Optional, Union

from jang.utils.conversions import JetModelBase
from jang.io import GW, NuDetector, SuperNuDetector, Parameters

matplotlib.use("Agg")


class ResDatabase:
    """Database containing all obtained results."""

    def __init__(self, filepath: Optional[str] = None, db: Optional[pd.DataFrame] = None):
        self._filepath = filepath
        self.db = db
        if filepath is not None and db is not None:
            logging.getLogger("jang").warning(
                "Both a file path and a DataFrame have been passed to Database, will use the database and ignore the file."
            )
        elif filepath is not None and os.path.isfile(filepath):
            self.db = pd.read_csv(filepath, index_col=0)

    def add_entry(
        self,
        detector: Union[NuDetector, SuperNuDetector],
        gw: GW,
        parameters: Parameters,
        limit_flux: float,
        limit_etot: float,
        limit_fnu: float,
        file_flux: str = None,
        file_etot: str = None,
        file_fnu: str = None,
        custom: Optional[dict] = None,
    ):
        """Adds new entry to the database."""
        sample_names = []
        if isinstance(detector, NuDetector):
            sample_names = [s.shortname for s in detector.samples]
        elif isinstance(detector, SuperNuDetector):
            sample_names = ["%s/%s" % (det.name, s.shortname) for det in detector.detectors for s in det.samples]
        else:
            raise TypeError("Unsupported object as detector.")

        dist = gw.samples.distance_5_50_95
        entry = {
            "Detector.name": detector.name,
            "Detector.samples": ",".join(sample_names),
            "Detector.error_acceptance": (
                ",".join([f"{e:.3e}" for e in np.sqrt(np.diag(detector.error_acceptance))])
                if parameters.apply_det_systematics
                else ""
            ),
            "Detector.results.nobserved": ",".join([f"{s.nobserved:d}" for s in detector.samples]),
            "Detector.results.nbackground": ",".join([f"{s.background}" for s in detector.samples]),
            "GW.catalog": gw.catalog,
            "GW.name": gw.name,
            "GW.mass1": gw.samples.masses[0],
            "GW.mass2": gw.samples.masses[1],
            "GW.type": gw.samples.type,
            "GW.distance_median": dist[1],
            "GW.distance_errorminus": dist[1] - dist[0],
            "GW.distance_errorplus": dist[2] - dist[1],
            "Parameters.systematics": "on" if parameters.apply_det_systematics else "off",
            "Parameters.systematics.ntoys": (
                parameters.ntoys_det_systematics if parameters.apply_det_systematics else ""
            ),
            "Parameters.energy_range": parameters.range_energy_integration,
            # "Parameters.neutrino_spectrum": parameters.spectrum,
            "Parameters.jet_model": parameters.jet.__repr__(),
            "Results.limit_flux": limit_flux,
            "Results.limit_etot": limit_etot,
            "Results.limit_fnu": limit_fnu,
            "Results.posterior_flux": file_flux,
            "Results.posterior_etot": file_etot,
            "Results.posterior_fnu": file_fnu,
        }
        if custom is not None:
            entry.update({f"Custom.{k}": v for k, v in custom.items()})
        newline = pd.DataFrame([entry])
        if self.db is None:
            self.db = newline
        else:
            self.db = pd.concat([self.db, newline], ignore_index=True)

    def save(self, filepath: Optional[str] = None):
        """Save the datavase to specified CSV or, by default, to the one defined when initialising the Database."""
        outfile = None
        if filepath is not None:
            outfile = filepath
        elif self._filepath is not None:
            outfile = self._filepath
        else:
            raise RuntimeError("No output file was provided.")
        self.sort()
        self.db.to_csv(outfile)

    def select_detector(self, det: Union[NuDetector, SuperNuDetector, str]):
        """Select a given detector and return a reduced database."""
        if isinstance(det, str):
            return ResDatabase(db=self.db[self.db["Detector.name"] == det])
        else:
            return ResDatabase(db=self.db[self.db["Detector.name"] == det.name])

    # def select_spectrum(self, spectrum: str):
    #     """Select a given neutrino spectrum."""
    #     return ResDatabase(db=self.db[self.db["Parameters.neutrino_spectrum"] == spectrum])

    def select_jetmodel(self, jetmodel: Union[JetModelBase, str]):
        """Select a given jet model."""
        if isinstance(jetmodel, str):
            return ResDatabase(db=self.db[self.db["Parameters.jet_model"] == jetmodel])
        else:
            return ResDatabase(db=self.db[self.db["Parameters.jet_model"] == jetmodel.__repr__()])

    def select_gwtype(self, gwtype: str):
        return ResDatabase(db=self.db[self.db["GW.type"] == gwtype])

    def select(
        self,
        det: Optional[Union[NuDetector, SuperNuDetector, str]] = None,
        # spectrum: Optional[str] = None,
        jetmodel: Optional[Union[JetModelBase, str]] = None,
        gwtype: Optional[str] = None,
    ):
        if det is None and jetmodel is None and gwtype is None:
            logging.getLogger("jang").info("No selection is applied, return the infiltered database.")
        db = copy.copy(self)
        if det is not None:
            db = db.select_detector(det)
        # if spectrum is not None:
        #     db = db.select_spectrum(spectrum)
        if jetmodel is not None:
            db = db.select_jetmodel(jetmodel)
        if gwtype is not None:
            db = db.select_gwtype(gwtype)
        return db

    def sort(self):
        self.db = self.db[sorted(self.db.columns)]
        self.db.sort_values(
            by=[
                "GW.catalog",
                "GW.name",
                "Parameters.jet_model",
            ],
            ascending=[True, True, True],
            na_position="first",
            inplace=True,
        )

    def plot_energy_vs_distance(self, outfile: str, cat: Optional[dict] = None):
        """Draw the distribution of the obtained limits, for a given selection of entries, as a function of luminosity distance."""

        plt.close("all")
        if cat is None:
            plt.errorbar(
                x=self.db["GW.distance_median"],
                y=self.db["Results.limit_etot"],
                xerr=[
                    self.db["GW.distance_errorminus"],
                    self.db["GW.distance_errorplus"],
                ],
                linewidth=0,
                elinewidth=1,
                marker="x",
            )
        else:
            for c, col, mar, lab in zip(cat["categories"], cat["colors"], cat["markers"], cat["labels"]):
                db_cat = self.db[self.db[cat["column"]] == c]
                plt.errorbar(
                    x=db_cat["GW.distance_median"],
                    y=db_cat["Results.limit_etot"],
                    xerr=[
                        db_cat["GW.distance_errorminus"],
                        db_cat["GW.distance_errorplus"],
                    ],
                    linewidth=0,
                    elinewidth=1,
                    color=col,
                    marker=mar,
                    label=lab,
                )
            plt.legend(loc="upper left", fontsize=17)
        plt.xlim((1e2, 2e4))
        plt.xlabel("distance [Mpc]", fontsize=16)
        plt.ylabel(r"$E^{90\%}_{tot,\nu}$ [erg]", fontsize=16)
        plt.xscale("log")
        plt.yscale("log")
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(outfile, dpi=300)

    def plot_fnu_vs_distance(self, outfile: str, cat: Optional[dict] = None):
        """Draw the distribution of the obtained limits, for a given selection of entries, as a function of luminosity distance."""

        plt.close("all")
        if cat is None:
            plt.errorbar(
                x=self.db["GW.distance_median"],
                y=self.db["Results.limit_fnu"],
                xerr=[
                    self.db["GW.distance_errorminus"],
                    self.db["GW.distance_errorplus"],
                ],
                linewidth=0,
                elinewidth=1,
                marker="x",
            )
        else:
            for c, col, mar, lab in zip(cat["categories"], cat["colors"], cat["markers"], cat["labels"]):
                db_cat = self.db[self.db[cat["column"]] == c]
                plt.errorbar(
                    x=db_cat["GW.distance_median"],
                    y=db_cat["Results.limit_fnu"],
                    xerr=[
                        db_cat["GW.distance_errorminus"],
                        db_cat["GW.distance_errorplus"],
                    ],
                    linewidth=0,
                    elinewidth=1,
                    color=col,
                    marker=mar,
                    label=lab,
                )
            plt.legend(loc="upper left", fontsize=17)
        plt.xlabel("distance [Mpc]", fontsize=16)
        plt.ylabel(r"$E^{90\%}_{tot,\nu}/E_{radiated}$", fontsize=16)
        plt.xscale("log")
        plt.yscale("log")
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(outfile, dpi=300)

    def plot_flux(self, outfile: str, cat: Optional[dict] = None):

        plt.close("all")
        plt.figure(figsize=(16, 6))
        min_flux, max_flux = np.inf, 0
        if cat is None:
            names = self.db["GW.name"]
            ii = range(len(self.db.index))
            plt.plot(
                ii,
                self.db["Results.limit_flux"],
                color="tab:blue",
                linewidth=0,
                marker="x",
                markersize=10,
            )
            min_flux = np.nanmin(self.db["Results.limit_flux"])
            max_flux = np.nanmax(self.db["Results.limit_flux"])
            plt.xticks(np.arange(len(names)), names, rotation=65, ha="right", fontsize=15)
        else:
            for c, col, mar, lab in zip(cat["categories"], cat["colors"], cat["markers"], cat["labels"]):
                db_cat = self.db[self.db[cat["column"]] == c]
                ii = db_cat.index.values
                plt.plot(
                    ii,
                    db_cat["Results.limit_flux"],
                    color=col,
                    linewidth=0,
                    marker=mar,
                    markersize=9,
                    label=lab,
                )
                min_flux = min(min_flux, np.nanmin(self.db["Results.limit_flux"]))
                max_flux = max(max_flux, np.nanmax(self.db["Results.limit_flux"]))
            plt.legend(loc="upper center", ncol=len(cat["categories"]))
            plt.xticks(
                self.db.index.values,
                self.db["GW.name"],
                rotation=65,
                va="top",
                ha="right",
                fontsize=15,
            )
        #
        plt.yscale("log")
        plt.ylabel(r"$E^2\dfrac{dn}{dE}$ [GeV cm$^{-2}$]", fontsize=16)
        plt.ylim((10 ** floor(np.log10(min_flux)), 10 ** ceil(np.log10(max_flux))))
        plt.grid(axis="y", which="major")
        plt.grid(axis="y", which="minor", linewidth=0.6)
        plt.grid(axis="x", which="major", linestyle="--")
        plt.tight_layout()
        plt.savefig(outfile, dpi=300)

    def plot_summary_observations(self, outfile: str, sample_colors: dict):

        names = self.db["GW.name"]
        nbkg = [[float(n.split(" ")[0]) for n in N.split(",")] for N in self.db["Detector.results.nbackground"]]
        nobs = [[int(float(n.split(" ")[0])) for n in N.split(",")] for N in self.db["Detector.results.nobserved"]]
        ngw = len(names)

        samples = [S.split(",") for S in self.db["Detector.samples"]]

        x = np.arange(ngw)
        yoffset = np.zeros(ngw)

        plt.close("all")
        plt.figure(figsize=(16, 6))
        for smp, col in sample_colors.items():
            idxs = [S.index(smp) if smp in S else None for S in samples]
            bkg = [nbkg[j][idxs[j]] if idxs[j] is not None else np.nan for j in range(ngw)]
            obs = [nobs[j][idxs[j]] if idxs[j] is not None else np.nan for j in range(ngw)]
            plt.bar(x, bkg, width=0.85, bottom=yoffset, color=col, alpha=0.5)
            plt.plot(x, obs, marker="*", linewidth=0, color=col)
            yoffset += np.array(bkg)

        plt.xticks(
            x,
            names,
            rotation=65,
            va="top",
            ha="right",
            fontsize=14,
        )
        plt.ylabel("Number of events")
        plt.yscale("log")
        plt.grid(axis="x")

        plt.tight_layout()
        plt.savefig(outfile, dpi=300)
