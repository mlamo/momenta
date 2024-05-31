"""Running an example analysis to do some tests."""

import tempfile
import unittest
import healpy as hp
import numpy as np

import jang.utils.conversions
import jang.utils.flux
from jang.io import GWDatabase, NuDetector, Parameters, ResDatabase
from jang.io.neutrinos import BackgroundFixed, EffectiveAreaBase
import jang.analysis.limits as limits
import jang.analysis.significance as significance
import jang.analysis.stacking as stacking


class EffectiveArea(EffectiveAreaBase):
    def __init__(self, norm):
        super().__init__()
        self.norm = norm

    def evaluate(self, energy):
        return self.norm


class TestExamples(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
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
        self.config_file = f"{self.tmpdir}/config.yaml"
        with open(self.config_file, "w") as f:
            f.write(config_str)
        #
        detector_str = """
            name: TestDet

            nsamples: 2
            samples:
              names: ["sampleA", "sampleB"]
              shortnames: ["A", "B"]
              energyrange: [0, 100]

            earth_location:
              latitude: 10.0
              longitude: 50.0
              units: deg

            errors:
              acceptance: 0.00
              acceptance_corr: 1
              background: 0.00
        """
        self.det_file = f"{self.tmpdir}/detector.yaml"
        with open(self.det_file, "w") as f:
            f.write(detector_str)
        self.accs = [np.ones(hp.nside2npix(8)), np.ones(hp.nside2npix(8))]
        #
        self.gwdb_file = "examples/input_files/gw_catalogs/database_example.csv"
        self.db_file = f"{self.tmpdir}/db.csv"

        # configuration
        self.pars = Parameters(self.config_file)
        self.pars.set_models(jang.utils.flux.FluxFixedPowerLaw(1, 100, 2), jang.utils.conversions.JetIsotropic())
        # GW database
        database_gw = GWDatabase(self.gwdb_file)
        database_gw.set_parameters(self.pars)
        self.gw = database_gw.find_gw("GW190412")
        # detector
        self.det = NuDetector(self.det_file)
        bkg = [BackgroundFixed(b) for b in [0.1, 0.3]]
        self.det.set_effective_areas([EffectiveArea(2), EffectiveArea(1)])
        self.det.set_observations([0, 0], bkg)

    def test_limits_nosyst(self):
        self.pars.apply_det_systematics = False
        self.pars.likelihood_method = "poisson"
        limits.get_limit_flux(self.det, self.gw, self.pars, f"{self.tmpdir}/flux")
        limits.get_limit_etot(self.det, self.gw, self.pars, f"{self.tmpdir}/etot")
        limits.get_limit_fnu(self.det, self.gw, self.pars, f"{self.tmpdir}/fnu")
        significance.compute_prob_null_hypothesis(self.det, self.gw, self.pars)

    def test_limits_wsyst(self):
        self.pars.apply_det_systematics = True
        self.pars.ntoys_det_systematics = 10
        self.pars.likelihood_method = "poisson"
        limits.get_limit_flux(self.det, self.gw, self.pars, f"{self.tmpdir}/flux")
        limits.get_limit_etot(self.det, self.gw, self.pars, f"{self.tmpdir}/etot")
        limits.get_limit_fnu(self.det, self.gw, self.pars, f"{self.tmpdir}/fnu")
        significance.compute_prob_null_hypothesis(self.det, self.gw, self.pars)

    def test_limits_pointsource(self):
        self.pars.apply_det_systematics = False
        self.pars.likelihood_method = "pointsource"
        limits.get_limit_flux(self.det, self.gw, self.pars, f"{self.tmpdir}/flux")
        limits.get_limit_etot(self.det, self.gw, self.pars, f"{self.tmpdir}/etot")
        limits.get_limit_fnu(self.det, self.gw, self.pars, f"{self.tmpdir}/fnu")

    def test_results_db(self):
        # make fake lkl files for etot and fnu (needed for stacking)
        x, y = np.logspace(*self.pars.range_etot), np.flipud(np.arange(self.pars.range_etot[-1]))
        np.save(f"{self.tmpdir}/etot", [x, y])
        x, y = np.logspace(*self.pars.range_fnu), np.flipud(np.arange(self.pars.range_fnu[-1]))
        np.save(f"{self.tmpdir}/fnu", [x, y])
        # save in database
        database_res = ResDatabase(self.db_file)
        database_res.add_entry(
            self.det,
            self.gw,
            self.pars,
            1,
            1e55,
            1,
            f"{self.tmpdir}/flux",
            f"{self.tmpdir}/etot",
            f"{self.tmpdir}/fnu",
            custom={"test": 0},
        )
        database_res.add_entry(
            self.det,
            self.gw,
            self.pars,
            2,
            2e55,
            2,
            f"{self.tmpdir}/flux",
            f"{self.tmpdir}/etot",
            f"{self.tmpdir}/fnu",
            custom={"test": 0},
        )
        database_res.save()
        # open database
        database_res = ResDatabase(self.db_file)
        with self.assertLogs(level="INFO"):
            database_res = database_res.select()
        database_res = database_res.select(det=self.det, jetmodel=self.pars.jet)
        # make plots
        cat = {
            "column": "GW.type",
            "categories": ["BBH", "BNS", "NSBH"],
            "labels": ["BBH", "BNS", "NSBH"],
            "colors": ["black", "blue", "orange"],
            "markers": ["+", "x", "^"],
        }
        database_res.plot_energy_vs_distance(f"{self.tmpdir}/eiso.png")
        database_res.plot_energy_vs_distance(f"{self.tmpdir}/eiso.png", cat=cat)
        database_res.plot_fnu_vs_distance(f"{self.tmpdir}/fnu.png")
        database_res.plot_fnu_vs_distance(f"{self.tmpdir}/fnu.png", cat=cat)
        database_res.plot_flux(f"{self.tmpdir}/flux.png")
        database_res.plot_flux(f"{self.tmpdir}/flux.png", cat=cat)
        database_res.plot_summary_observations(
            f"{self.tmpdir}/obs.png", {s.shortname: "black" for s in self.det.samples}
        )
        #
        stacking.stack_events(database_res, self.pars)
        with self.assertLogs(level="ERROR"):
            stacking.stack_events_listgw(
                database_res, ["GW190412", "missing_ev"], self.pars
            )
        stacking.stack_events_weightedevents(
            database_res,
            {"GW190412": 1},
            self.pars,
            outfile=f"{self.tmpdir}/stacking.png",
        )
        with self.assertLogs(level="ERROR"):
            stacking.stack_events_weightedevents(
                database_res,
                {"GW190412": 1, "missing_ev": 0.5},
                self.pars,
                outfile=f"{self.tmpdir}/stacking.png",
            )
