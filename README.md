# Joint Analysis of Neutrinos and Gravitational waves

![logo](https://github.com/mlamo/jang/blob/main/doc/logo.png?raw=true)

[![tests](https://github.com/mlamo/jang/actions/workflows/tests.yml/badge.svg)](https://github.com/mlamo/jang/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/mlamo/jang/branch/main/graph/badge.svg?token=PVBSZ9P7TR)](https://codecov.io/gh/mlamo/jang)

# Method

This framework is using a Bayesian approach to convert observations from neutrino telescopes to constraints on the neutrino emission from transient astrophysical sources. It is aimed to combine observations from several neutrino samples into a single set of constraints.

## Inputs

The input ingredients from neutrino side are:
* number of observed events in each neutrino sample → $N_{s}$
* number of expected background events in each neutrino sample (see below) → $B_{s}$
* detector effective area as a function of neutrino energy and direction → $A_{\rm eff,s}(E,\Omega)$
* OPTIONAL: list of observed events with their time, direction, and energy → $\{\text{ev}_{i,s}\}$
* OPTIONAL: other instrumental response functions such as angular and energy pdf for signal and background hypotheses → $S_{\rm ang,s}$, $B_{\rm ang,s}$, $S_{\rm ene,s}$, $B_{\rm ene,s}$

The inputs from transient source are:
* localisation of the source (see below) → $\Omega_{\rm src}$
* OPTIONAL: time of the event
* OPTIONAL: other information that may be relevant for astrophysical interpretations (lunimosity distance, redshift...)

Other general inputs:
* assumed neutrino energy spectrum which may include several components → $F(E) = \sum_{i=1}^{N_c} \phi_i \times f_i(E)$
* priors on the flux normalisation → $\pi(\phi_i)$ (can be uniform in linear/log scale, etc...)
* OPTIONAL: assumed jet structure

### Expected background

The expected background will be incorporated as a prior in the analysis. Three scenarii are implemented:
* fixed background: $\pi(B_s \vert B_{s,0}) = \delta(B_s - B_{s,0})$
 → user should provide $B_{s,0}$
* Gaussian background: $\pi(B_s \vert B_{s,0}, \sigma_{B_s}) = \textrm{Gaus}(B_s; \mu=B_{s,0}, \sigma=\sigma_{B_s})$
 → user should provide mean $B_{s,0}$ and error $\sigma_{B_s}$
* background from ON/OFF measurement: $\pi(B_s \vert N^{\rm off}_s, \alpha^{\rm OFF/ON}_s) = \textrm{Poisson}(N_{\rm OFF}; B_{s} \times \alpha^{\rm OFF/ON}_s)$
 → user should provide observed number of events in OFF region $N^{\rm off}_s$ and ratio between OFF and ON $\alpha^{\rm OFF/ON}_s)$

The backgrounds in the different samples are assumed to be uncorrelated.

### Additional uncertainties

Additional uncertainties on the effective area may be incorporated as prior. For simplification, this term will be neglected in the following.

### Source localisation

Different source types may be considered, but the two already implemented are:
* using GW posterior samples describing the localisation of the source
* fixed equatorial coordinates
Generally, these are represented by a prior $\pi(\Omega_{\rm src})$ (that is trivial for the second case).

## Likielihood for one given sample

The likelihood may be defined for two different cases:
* if we want to perform a simple cut&count search (cc), the likelihood will simply be a Poisson term
* if we want to perform a point-source-like analysis (ps), the likelihood will incorporate both a Poisson term and the angualr/energy pdfs

In both case, we need to define the expected number of signal events $N_{\rm sig,s}$ that depends on all the relevant inputs:
$$N_{\rm sig,s}(\{\phi_i\}, \Omega_{\rm src}) = \sum_i \phi_i \int A_{\rm eff,s}(E,\Omega_{\rm src}) \times f_i(E) {\rm d}E$$

We can then write the two likelihoods:

$$\mathcal{L}_{cc}(N_{s} \vert \{\phi_i\}, B_s, \Omega_{\rm src}) = \textrm{Poisson}\left(N_s; B_s + N_{\rm sig,s}(\{\phi_i\}, \Omega_{\rm src})\right)$$
$$\mathcal{L}_{ps}(N_{s}, \{\text{ev}_{j,s}\} \vert \{\phi_i\}, B_s, \Omega_{\rm src}) = \mathcal{L}_{cc}(N_{s}, \vert \{\phi_i\}, B_s, \Omega_{\rm src}) \times \prod_{j \in s} \dfrac{B_s p_{\rm bkg}(\text{ev}_j) + N_{\rm sig,s}(\{\phi_i\}, \Omega_{\rm src}) p_{\rm sig}(\text{ev}_j \vert \{\phi_i\}, \Omega_{\rm src})}{B_s + N_{\rm sig,s}(\{\phi_i\}, \Omega_{\rm src})}

In the point-source case, $p_{\rm bkg}$ and $p_{\rm sig}$ are the probabilities for event $j$ to be background or signal. These are built from the instrumental response functions. For instance, if we just incorporate the point-spread function, we have:
* $p_{\rm bkg}(\text{ev}_j) = f(\Omega_j)$ depends solely on event direction $\Omega_j$ and described how likely this direction is in the background hypothesis, the function is normalized such that $\int f(\Omega) d\Omega = 1$
* $p_{\rm sig}(\text{ev}_j) = g(\Omega_j, \sigma_j, \Omega_{\rm src})$ depends on event direction $\Omega_j$, uncertainty on this direction, and source direction. The function is normalized such that $\int g(\Omega, \sigma_j, \Omega_{\rm src}) d\Omega = 1$ for any value of $\sigma_j$ and $\Omega_{\rm src}$.

## Posterior probability

Generally, we define the posterior probability as the product of the contribution of the different neutrino samples and all the priors:

$$P(\{phi_i\}, \{B_s\}, \Omega_{\rm src}, \ldots) = \prod_s \mathcal{L}(N_{s}, \{\text{ev}_{j,s}\} \vert \{\phi_i\}, B_s, \Omega_{\rm src}) \times \prod_s \pi(B_s) \times \pi(\Omega_{\rm src}) \times \pi(\{\phi_i\})$$


# Installation

* Clone the repository: ``git clone https://github.com/mlamo/jang.git``
* Install the package: ``cd jang && pip install -e .``

# Step-by-step usage

## Parameters

* Create/use a YAML file with all needed parameters (example: ``examples/input_files/config.yaml``)
* Load the parameters:
```python
from jang.io import Parameters
pars = Parameters("examples/parameter_files/path_to_yaml_file")
```

* Select the neutrino spectrum and jet model:
```python
pars.set_models("x**-2", jang.utils.conversions.JetIsotropic())
```
(the list of available jet models is available in ``jang/conversions.py``)

## Detector information
   
* Create/use a YAML file with all relevant information (examples in ``examples/input_files/DETECTORNAME/detector.yaml``)
* Create a new detector object:
```python
from jang.io import NuDetector
det = NuDetector(path_to_yaml_file)
```

* Acceptances have to be defined for the spectrum to consider:
   * if they already exist in npy or ndarray format (one for each sample), they can directly be loaded:
   ```python
   det.set_acceptances(npy_path_or_ndarray, spectrum="x**-2")
   ```

   * otherwise, it can be estimated using an object of a class derived from EffectiveAreaBase, as illustrated for Super-Kamiokande in ``examples/superkamiokande.py``

* Any observation can be set with the following commands, where the two arguments are arrays with one entry per sample:
```python
from jang.io.neutrinos import BackgroundFixed
# different background models are available: BackgroundFixed(b0), BackgroundGaussian(b0, deltab), BackgroundPoisson(Noff, Nregionsoff)
bkg = [BackgroundFixed(0.51), BackgroundFixed(0.12)]
det.set_observations(n_observed=[0,0], background=bkg)
```

## GW information

* GW database can be easily imported using an existing csv file (see e.g., ``examples/input_files/gw_catalogs/database_example.csv``):
```python
from jang.io import GWDatabase
database_gw = GWDatabase(path_to_csv)
```

* An event can be extracted from it:
```python
gw = database_gw.find_gw(name_of_gw, pars)
```

* Alternatively, one may specify directly the files for a given event
```python
from jang.io import GW
gw = GW(name, path_to_fits_files, path_to_hdf5_file)
```

## Compute limits

* Limit on the incoming neutrino flux (where the last optional argument is the local path -without extension- where the posterior could be saved in npy format):
```python
import jang.analysis.limits as limits
limits.get_limit_flux(det, gw, pars, path_to_file)
```

* Same for the total energy emitted in neutrinos:
```python
limits.get_limit_etot(det, gw, pars, path_to_file)
```

* Same for the ratio fnu=E(tot)/E(rad,GW):
```python
limits.get_limit_fnu(det, gw, pars, path_to_file)
```

## Results database
   
* Create/open the database:
``` python
import jang.io.ResDatabase
database_res = ResDatabase(path_to_csv)
```

* Add new entries in the database:
```python
database_res.add_entry(det, gw, pars, limit_flux, limit_etot, limit_fnu, path_to_flux, path_to_etot, path_to_fnu)
```

* Save the database:
```python
database_res.save()
```

# Full examples

Some full examples are available in `examples/`:
* `superkamiokande.py` provides a full example using Super-Kamiokande public effective areas from [Zenodo](https://zenodo.org/records/4724823) and expected background rates from [Astrophys.J. 918 (2021) 2, 78](https://doi.org/10.3847/1538-4357/ac0d5a).
* `full_example.ipynb` provides a step-by-step example to get Super-Kamiokande/ANTARES sensitivities and perform a combination. The ANTARES acceptance are rough estimates from [JCAP 04 (2023) 004](https://arxiv.org//abs/2302.07723).