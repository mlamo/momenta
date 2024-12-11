"""
    Copyright (C) 2024  Mathieu Lamoureux

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import contextlib
import logging
import os

from momenta.io import NuDetectorBase, Transient, Parameters
from momenta.stats.model import ModelNested

import ultranest


logger = logging.getLogger("ultranest")
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.WARNING)


@contextlib.contextmanager
def redirect_stdout(dest_filename):
    """
    A context manager to temporarily redirect stdout or stderr e.g.:
    with sys.stdout_redirected(sys.stderr, os.devnull):
        ...
    """
    try:
        old = os.dup(1), os.dup(2)
        dest_file = open(dest_filename, "w")
        os.dup2(dest_file.fileno(), 1)
        os.dup2(dest_file.fileno(), 1)
        yield
    finally:
        os.dup2(old[0], 1)
        os.dup2(old[1], 2)
        dest_file.close()


def run_ultranest(
    detector: NuDetectorBase,
    src: Transient,
    parameters: Parameters,
    precision_dlogz: float = 0.3,
    precision_dKL: float = 0.1,
):
    # two classes have things set after the constructor that will be needed here:
    detector.validate() # detector is sufficiently described
    parameters.validate() # all needed parameters e.g. flux model
    model = ModelNested(detector, src, parameters)
    sampler = ultranest.ReactiveNestedSampler(model.param_names, model.loglike, model.prior, vectorized=True)
    result = sampler.run(show_status=False, viz_callback=False, dlogz=precision_dlogz, dKL=precision_dKL)

    result["samples"] = {k: v for k, v in zip(model.param_names, result["samples"].transpose()) if k.startswith("flux") or k == "itoy"}
    result["weighted_samples"]["points"] = {
        k: v for k, v in zip(model.param_names, result["weighted_samples"]["points"].transpose()) if k.startswith("flux") or k == "itoy"
    }
    return model, result
