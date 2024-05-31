from astropy.units import deg
import healpy as hp
import numpy as np
from typing import Iterable, Union
import unittest

from jang.io.neutrinos import (
    infer_uncertainties,
    BackgroundFixed,
    BackgroundGaussian,
    BackgroundPoisson,
    NuDetector,
    EffectiveAreaBase,
    NuSample,
    SuperNuDetector,
    ToyNuDet,
)


class TestSample(unittest.TestCase):
    def setUp(self):
        self.s1 = NuSample("sample")
        self.s1.set_observations(0, BackgroundFixed(0.5))
        self.s1.set_energy_range(1, 1000)

    def test_members(self):
        self.assertEqual(self.s1.shortname, "sample")
        self.assertEqual(self.s1.log10_energy_range, (0, 3))
        self.assertEqual(self.s1.log_energy_range, (np.log(1), np.log(1000)))
        self.assertEqual(self.s1.nobserved, 0)
        self.assertEqual(self.s1.background.nominal, 0.5)

    def test_background(self):
        print(BackgroundFixed(1))
        print(BackgroundGaussian(1, 0.1))
        print(BackgroundPoisson(10, 10))


class TestDetector(unittest.TestCase):
    def setUp(self):
        self.dict_det1 = {
            "name": "Test",
            "nsamples": 1,
            "samples": {"names": ["smp"], "energyrange": [0, 1]},
            "earth_location": {"latitude": 0, "longitude": 0, "units": "deg"},
            "errors": {"acceptance": 0, "acceptance_corr": 0, "background": 0},
        }
        self.dict_det2 = {
            "name": "Test2",
            "nsamples": 4,
            "samples": {
                "names": ["sample1", "sample2", "sample3", "sample4"],
                "shortnames": ["s1", "s2", "s3", "s4"],
                "energyrange": [[0, 1], [0, 1], [0, 1], [0, 1]],
            },
            "earth_location": {"latitude": 10.0, "longitude": 5.0, "units": "deg"},
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
        self.assertEqual(self.d1.samples[0].shortname, "smp")

    def test_exceptions(self):
        self.dict_det1["samples"]["energyrange"] = 0
        with self.assertRaises(RuntimeError):
            NuDetector(self.dict_det1)
        with self.assertRaises(TypeError):
            NuDetector(0)
        with self.assertRaises(RuntimeError):
            self.d1.set_observations([0, 0], [BackgroundFixed(0)])
        with self.assertRaises(RuntimeError):
            self.d1.set_observations([0], [BackgroundFixed(0), BackgroundFixed(0)])

    def test_toys(self):
        with self.assertRaises(RuntimeError):
            self.d1.prepare_toys(0)
        t = self.d2.prepare_toys(0)
        self.assertEqual(len(t), 1)
        t = self.d2.prepare_toys(500)
        self.assertEqual(len(t), 500)

    def test_superdetector(self):
        sd = SuperNuDetector("SD")
        sd.add_detector(self.d1)
        sd.add_detector(self.d2)
        with self.assertLogs(level="ERROR"):
            sd.add_detector(self.d2)
        with self.assertRaises(RuntimeError):
            sd.prepare_toys(500)
        self.d1.set_observations([0], [BackgroundFixed(0)])
        self.assertEqual(sd.nsamples, 5)
        self.assertEqual(len(list(sd.samples)), 5)
        sd.prepare_toys(0)
        sd.prepare_toys(500)


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

    def test_toyresult(self):
        t = ToyNuDet([0, 1], [0.5, 1.5], [1, 1])
        self.assertEqual(
            t.__str__(),
            "ToyNuDet: n(observed)=[0 1], n(background)=[0.5 1.5], var(acceptance)=[1 1], events=None",
        )
