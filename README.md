# Multi-Observations Multi-Energy Neutrino Transient Analysis

![logo](https://github.com/mlamo/momenta/blob/main/doc/logo.png?raw=true)

[![tests](https://github.com/mlamo/momenta/actions/workflows/tests.yml/badge.svg)](https://github.com/mlamo/momenta/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/mlamo/momenta/branch/main/graph/badge.svg?token=PVBSZ9P7TR)](https://codecov.io/gh/mlamo/momenta)


## Installation

* Clone the repository: ``git clone https://github.com/mlamo/momenta.git``
* Install the package: ``cd momenta && pip install -e .``

## Step-by-step usage

### Parameters

* Create/use a YAML file with all needed parameters (example: ``examples/input_files/config.yaml``)
* Load the parameters:
```python
from momenta.io import Parameters
pars = Parameters("examples/parameter_files/path_to_yaml_file")
```

* Select the neutrino spectrum and eventually jet model:
```python
import momenta.utils.flux
import momenta.utils.conversions
flux = momenta.utils.flux.FluxFixedPowerLaw(1, 1e6, 2, eref=1)
jet = momenta.utils.conversions.JetVonMises(np.deg2rad(10))
pars.set_models(flux, jet=jet)
```
(the list of available jet models is available in ``src/momenta/utils/conversions.py``)

### Detector information
   
* Create/use a YAML file with all relevant information (examples in ``examples/input_files/DETECTORNAME/detector.yaml``)
* Create a new detector object:
```python
from momenta.io import NuDetector
det = NuDetector(path_to_yaml_file)
```

* Effective areas have to be defined for each neutrino sample. You can find basic classes in ``src/momenta/io/neutrinos`` and implementation examples in ``examples/``
```python
det.set_effective_areas([effarea1, effarea2, ...])
```

* Any observation can be set with the following commands, where the two arguments are arrays with one entry per sample:
```python
from momenta.io.neutrinos import BackgroundFixed
# different background models are available: BackgroundFixed(b0), BackgroundGaussian(b0, deltab), BackgroundPoisson(Noff, alphaoffon)
bkg = [BackgroundFixed(0.51), BackgroundFixed(0.12)]
det.set_observations(n_observed=[0,0], background=bkg)
```

### Source information

* GW database can be easily imported using an existing csv file (see e.g., ``examples/input_files/gw_catalogs/database_example.csv``):
```python
from momenta.io import GWDatabase
database_gw = GWDatabase(path_to_csv)
```

* A GW event can be extracted from it:
```python
gw = database_gw.find_gw(name_of_gw, pars)
```

* For point sources, one may use:
```python
from momenta.io.transient import PointSource
ps = PointSource(ra_deg=123.45, dec_deg=67.89, name="srcABC")
```

### Obtain results

* Run the nested sampling algorithm:
```python
from momenta.stats.run import run_ultranest
model, result = run_ultranest(det, gw, pars)
```

* Look to posterior samples:
```python
print("Available parameters:", model.param_names)
print("Samples:", result["samples"])
```

* Obtain X% upper limits:
```python
limits = get_limits(result["samples"], model)
print("Limit on the flux normalisation of the first component", limits["flux0_norm"])
```

## Full examples

Some full examples are available in `examples/`:
* `superkamiokande.py` provides a full example using Super-Kamiokande public effective areas from [Zenodo](https://zenodo.org/records/4724823) and expected background rates from [Astrophys.J. 918 (2021) 2, 78](https://doi.org/10.3847/1538-4357/ac0d5a).
* `full_example.ipynb` provides a step-by-step example to get sensitivities and perform a combination of different detectors.