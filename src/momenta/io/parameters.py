"""Class to hold general analysis parameters."""

import numpy as np
import os
import yaml
from typing import Optional

from momenta.utils.flux import FluxBase
from momenta.utils.conversions import JetModelBase


class Parameters:
    def __init__(self, file: Optional[str] = None):
        self.file = None
        self.flux = None
        self.jet = None
        if file is not None:
            assert os.path.isfile(file)
            self.file = file
            with open(self.file, "r") as f:
                params = yaml.safe_load(f)
            # analysis parameters
            self.nside = params.get("skymap_resolution")
            self.apply_det_systematics = bool(params["detector_systematics"])
            self.likelihood_method = params["analysis"]["likelihood"]
            # signal priors
            self.prior_normalisation = params["analysis"]["prior_normalisation"]["type"]
            self.prior_normalisation_range = params["analysis"]["prior_normalisation"]["range"]
            # GW parameters
            if "gw" in params and "sample_priorities" in params["gw"]:
                self.gw_posteriorsamples_priorities = params["gw"]["sample_priorities"]
            else:
                self.gw_posteriorsamples_priorities = [
                    "PublicationSamples",
                    "C01:IMRPhenomXPHM",
                    "IMRPhenomXPHM",
                    "C01:IMRPhenomPv3HM",
                    "IMRPhenomPv3HM",
                    "C01:IMRPhenomPv2",
                    "IMRPhenomPv2",
                    "C01:IMRPhenomNSBH:HighSpin",
                    "IMRPhenomNSBH:HighSpin",
                    "C01:IMRPhenomNSBH:LowSpin",
                    "IMRPhenomNSBH:LowSpin",
                    "C01:Mixed",
                    "Mixed",
                ]

    def set_models(self, flux: FluxBase = None, jet: JetModelBase = None):
        """Set the neutrino flux model and jet model."""
        if flux is not None:
            self.flux = flux
        if jet is not None:
            self.jet = jet

    @property
    def str_filename(self):
        """Get the representation of the parameters in string format for suffixing filenames."""
        str_model = []
        if self.flux is not None:
            str_model.append(str(self.flux))
        if self.jet is not None:
            str_model.append(self.jet.str_filename)
        return "_".join(str_model)

    def get_searchregion_gwfraction(self) -> float:
        spl = self.search_region.split("_")
        if len(spl) >= 2 and spl[0] == "region":
            return float(spl[1]) / 100
        if len(spl) == 1 and spl[0] == "bestfit":
            return 0
        if len(spl) == 1 and spl[0] == "fullsky":
            return None
        return None

    def get_searchregion_iszeroincluded(self) -> bool:
        """Returns True if the pixels with zero acceptance should be included."""
        spl = self.search_region.split("_")
        if spl[-1] == "excludezero":
            return False
        return True
