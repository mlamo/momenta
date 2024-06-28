import numpy as np
import unittest

from momenta.io.neutrinos import (
    infer_uncertainties,
    BackgroundFixed,
    BackgroundGaussian,
    BackgroundPoisson,
    NuDetector,
    NuEvent,
    NuSample,
    SuperNuDetector,
)
import momenta.stats.pdfs as pdf
import momenta.utils.flux as flux


class EnergySignal(pdf.EnergySignal):
    def __init__(self):
        super().__init__()
    def __call__(self, evt, flux):
        return [1/(np.log10(c.emax) - np.log10(c.emin)) if c.emin <= evt.energy <= c.emax else 0 for c in flux.components]


class TestSample(unittest.TestCase):
    def setUp(self):
        self.s1 = NuSample("sample")
        self.s1.set_observations(0, BackgroundFixed(0.5))
        self.s1.set_pdfs(
            sig_ang=pdf.VonMisesSignal(),
            sig_ene=EnergySignal(),
            bkg_ang=pdf.AngularBackground(lambda ra,dec,energy: 1),
            bkg_ene=pdf.EnergyBackground(lambda ra,dec,energy: 1/energy)
        )

    def test_members(self):
        self.assertEqual(self.s1.name, "sample")
        self.assertEqual(self.s1.nobserved, 0)
        self.assertEqual(self.s1.background.nominal, 0.5)

    def test_background(self):
        self.assertEqual(BackgroundFixed(1).nominal, 1)
        self.assertEqual(BackgroundGaussian(1, 0.1).nominal, 1)
        self.assertEqual(BackgroundPoisson(10, 10).nominal, 1)
        
    def test_pdfs(self):
        f = flux.FluxFixedPowerLaw(1, 100, 2)
        ev = NuEvent(ra=0, dec=0, sigma=0.01, energy=10)
        ra_src, dec_src = 0, 0
        #
        p_sig = self.s1.compute_event_probability([1.0], 0.0, ev, ra_src, dec_src, f)
        p_angsig = 0.5/np.pi/ev.sigma**2
        p_enesig = 1/2
        self.assertEqual(p_sig, p_angsig*p_enesig)
        #
        p_bkg = self.s1.compute_event_probability([0.0], 1.0, ev, ra_src, dec_src, f)
        self.assertEqual(p_bkg, 1/ev.energy)    


class TestDetector(unittest.TestCase):
    def setUp(self):
        self.dict_det1 = {
            "name": "Test",
            "nsamples": 1,
            "samples": ["sample1"],
            "errors": {"acceptance": 0, "acceptance_corr": 0, "background": 0},
        }
        self.dict_det2 = {
            "name": "Test2",
            "nsamples": 4,
            "samples": ["sample1", "sample2", "sample3", "sample4"],
            "errors": {"acceptance": 0.40, "acceptance_corr": 1, "background": 0.40},
        }
        self.d1 = NuDetector(self.dict_det1)
        self.d2 = NuDetector(self.dict_det2)
        self.d2.set_observations(
            [0, 0, 0, 0],
            [
                BackgroundFixed(0.3),
                BackgroundGaussian(0.1, 0.01),
                BackgroundGaussian(0.1, 0.02),
                BackgroundPoisson(3, 10),
            ],
        )

    def test_members(self):
        self.assertEqual(self.d1.nsamples, 1)
        self.assertEqual(self.d2.nsamples, 4)
        self.assertEqual(self.d1.name, "Test")
        self.assertEqual(len(self.d2.samples), 4)
        self.assertEqual(self.d1.samples[0].name, "sample1")

    def test_exceptions(self):
        with self.assertRaises(TypeError):
            NuDetector(0)
        with self.assertRaises(RuntimeError):
            self.d1.set_observations([0, 0], [BackgroundFixed(0)])
        with self.assertRaises(RuntimeError):
            self.d1.set_observations([0], [BackgroundFixed(0), BackgroundFixed(0)])

    def test_superdetector(self):
        sd = SuperNuDetector("SD")
        sd.add_detector(self.d1)
        sd.add_detector(self.d2)
        with self.assertLogs(level="ERROR"):
            sd.add_detector(self.d2)
        self.d1.set_observations([0], [BackgroundFixed(0)])
        self.assertEqual(sd.nsamples, 5)
        self.assertEqual(len(list(sd.samples)), 5)


class TestOther(unittest.TestCase):
    def test_infer_uncertainties(self):
        self.assertTrue(
            np.array_equal(infer_uncertainties(0, 2), np.array([[0, 0], [0, 0]]))
        )
        self.assertTrue(
            np.array_equal(infer_uncertainties([1, 2], 2), np.array([[1, 0], [0, 4]]))
        )
        self.assertTrue(
            np.array_equal(
                infer_uncertainties([1, 2], 2, 1), np.array([[1, 2], [2, 4]])
            )
        )
        self.assertTrue(
            np.array_equal(
                infer_uncertainties([[1, 0], [0, 4]], 2), np.array([[1, 0], [0, 4]])
            )
        )
        self.assertIsNone(infer_uncertainties(None, 1))
        with self.assertRaises(RuntimeError):
            infer_uncertainties([0, 0, 0], 2)

