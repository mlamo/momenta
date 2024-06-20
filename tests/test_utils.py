import healpy as hp
import numpy as np
import tempfile
import unittest

from jang.io import Parameters, GWDatabase
import jang.utils.conversions
import jang.stats


class TestJetModels(unittest.TestCase):
    def test_isotropic(self):
        jet = jang.utils.conversions.JetIsotropic()
        jet.etot_to_eiso(0)
        print(jet, jet.str_filename)

    def test_vonmises(self):
        jet = jang.utils.conversions.JetVonMises(np.inf)
        jet.etot_to_eiso(0)
        jet.etot_to_eiso(0.5)
        print(jet, jet.str_filename)
        jet = jang.utils.conversions.JetVonMises(0.1)
        jet.etot_to_eiso(0)
        jet.etot_to_eiso(0.5)
        print(jet, jet.str_filename)
        jet = jang.utils.conversions.JetVonMises(0.1, with_counter=True)
        jet.etot_to_eiso(0)
        jet.etot_to_eiso(0.5)
        print(jet, jet.str_filename)

    def test_rectangular(self):
        jet = jang.utils.conversions.JetRectangular(np.inf)
        jet.etot_to_eiso(0)
        jet.etot_to_eiso(0.5)
        print(jet, jet.str_filename)
        jet = jang.utils.conversions.JetRectangular(0.1)
        jet.etot_to_eiso(0)
        jet.etot_to_eiso(0.5)
        print(jet, jet.str_filename)
        jet = jang.utils.conversions.JetRectangular(0.1, with_counter=True)
        jet.etot_to_eiso(0)
        jet.etot_to_eiso(0.5)
        print(jet, jet.str_filename)

    def test_list(self):
        jang.utils.conversions.list_jet_models()


class TestGW(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        config_str = """
            skymap_resolution: 8
            detector_systematics: 0

            analysis:
                likelihood: poisson
                prior_normalisation:
                    type: flat-linear
                    range: [0.0, 10000000000]
        """
        self.config_file = f"{self.tmpdir}/config.yaml"
        with open(self.config_file, "w") as f:
            f.write(config_str)
        self.pars = Parameters(self.config_file)
        self.dbgw = GWDatabase("examples/input_files/gw_catalogs/database_example.csv")
        self.dbgw.set_parameters(self.pars)
        self.gw = self.dbgw.find_gw("GW190412")
        self.tmpdir = tempfile.mkdtemp()

    def test_skymap(self):
        self.assertTrue(np.all(self.gw.fits.get_signal_region(8, None) == np.arange(hp.nside2npix(8))))
        self.assertTrue(np.all(self.gw.fits.get_signal_region(8, 0.90) == [163, 131, 164]))
        self.assertTrue(np.isclose(self.gw.fits.get_ra_dec_bestfit(8)[1], 35.68533471265204))

    def test_database(self):
        emptydb = GWDatabase()
        emptydb.add_entry("ev", "", "")
        with self.assertRaises(RuntimeError):
            emptydb.save()
        emptydb = GWDatabase(f"{self.tmpdir}/db.csv")
        emptydb.add_entry("ev", "", "")
        emptydb.save(f"{self.tmpdir}/db.csv")
        emptydb.save()
        #
        with self.assertRaises(RuntimeError):
            self.dbgw.find_gw("missing_ev")
        self.dbgw.list_all()
        self.dbgw.list("BBH", 0, 1000)
        self.dbgw.list("BNS", 1000, 0)
        self.dbgw.add_entry("ev", "", "")
        self.dbgw.save(f"{self.tmpdir}/db.csv")


class TestParameters(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        config_str = """
            analysis:
              nside: -1
              apply_det_systematics: 0
              ntoys_det_systematics: 0
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
