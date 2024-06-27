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

$$\mathcal{L}_{ps}(N_{s}, \{\text{ev}_{j,s}\} \vert \{\phi_i\}, B_s, \Omega_{\rm src}) = \mathcal{L}_{cc}(N_{s}, \vert \{\phi_i\}, B_s, \Omega_{\rm src}) \times \prod_{j \in s} \dfrac{B_s p_{\rm bkg}(\text{ev}_j) + N_{\rm sig,s}(\{\phi_i\}, \Omega_{\rm src}) p_{\rm sig}(\text{ev}_j \vert \{\phi_i\}, \Omega_{\rm src})}{B_s + N_{\rm sig,s}(\{\phi_i\}, \Omega_{\rm src})}$$

In the point-source case, $p_{\rm bkg}$ and $p_{\rm sig}$ are the probabilities for event $j$ to be background or signal. These are built from the instrumental response functions. For instance, if we just incorporate the point-spread function, we have:
* $p_{\rm bkg}(\text{ev}_j) = a(\Omega_j)$ depends solely on event direction $\Omega_j$ and described how likely this direction is in the background hypothesis.
* $p_{\rm sig}(\text{ev}_j) = a(\Omega_j, \sigma_j, \Omega_{\rm src})$ depends on event direction $\Omega_j$, uncertainty on this direction, and source direction. 
These functions are normalized such that $\int a(\Omega, \ldots) d\Omega = 1$.

## Posterior probability

Generally, we define the posterior probability distribution function as the product of the contribution of the different neutrino samples and all the priors:

$$P(\{\phi_i\}, \{B_s\}, \Omega_{\rm src} \vert \ldots) = \prod_s \mathcal{L}(N_{s}, \{\text{ev}_{j,s}\} \vert \{\phi_i\}, B_s, \Omega_{\rm src}) \times \prod_s \pi(B_s) \times \pi(\Omega_{\rm src}) \times \pi(\{\phi_i\})$$

We can eventually integrate over all nuisance parameters to get the marginalised posterior:

$$P_{\rm marg}(\{\phi_i\}) = \idotsint P(\{\phi_i\}, \{B_s\}, \Omega_{\rm src} \vert \ldots) {\rm d}\Omega_{\rm src} \prod_s {\rm d}B_s$$

From that, one may trivially extract flux upper limits, best-fit, contour plots...

## Bayes factors

Another feature of Bayesian analyses is the possibility to extract Bayes factor that plays the role of significance/p-values in frequentist approaches. In this case, we do not care about the extraction of upper limits but simply want to compare different models and see which one the data may favour.

Let's consider two hypotheses:
* $H_0$: there is only background ($N_{\rm sig,s} = 0$ for all $s$)
* $H_1$: there is both background and signal contributions with $F(E) = \phi (E/{\rm GeV})^{-\Gamma}$

We may then compute the related Bayes evidence starting from the posterior probability:
* For $H_0$, we just need to integrate over nuisance parameters as no source is involved
$$E_0 = \idotsint P(\{B_s\} \vert \ldots) \prod_s {\rm d}B_s$$
* For $H_1$, we need to integrate over all possible source parameters
$$E_1 =  \idotsint P(\phi, \{B_s\}, \Omega_{\rm src} \vert \ldots) {\rm d}\Omega_{\rm src} \prod_s {\rm d}B_s {\rm d}\phi$$

The Bayes factor is then naively defined as:
$$B^{\rm naive}_{10} = E_1 / E_0$$

However, when using noninformative priors on source parameters (such as flat ones $\pi(\phi) = 1/C$ for $0 \leq \phi < C$), the Bayes factor is defined up to a constant. Let's spell the simple cut-and-count approach with one sample, $N=0$, $A=\int A_{\rm eff,s}(E,\Omega_{\rm src}) E^{-\Gamma} {\rm d}E$, fixed background $B$, and fixed source position:

$$B^{\rm naive}_{10} = \dfrac{\int\textrm{Poisson}(0, B + \phi A(\Omega_{\rm src})) \times \pi(\phi) {\rm d}\phi}{\textrm{Poisson}(0, B)} = (1/C) \times \int_0^{C} e^{-\phi A(\Omega_{\rm src})} {\rm d}\phi = \dfrac{1-e^{-C A(\Omega_{\rm src})}}{C A(\Omega_{\rm src})}$$

Several approaches are available to correct for this. One of those is the usage of Arithmetic Intrinsic Bayes Factor (AIBF) where we use minimal training samples that cannot discriminate between the two models to compute a correction:

$$B^{\rm AI}_{10}^{\rm data} = B^{\rm naive}_{10}^{\rm data} \times (B^{\rm naive}_{10}^{\text{minimal set}})^{-1}$$

## Implementation

The basic need for the implementation of the Bayesian analysis is to be able to sample the posterior distribution. With this, if is then possible to:
* marginalise over nuisance parameters to get the marginalised posterior and compute constraints on flux parameters
* derive also constrained on other source parameters such as the total energy emitted in neutrinos assuming isotropic emission
* marginalise over nuisance+source parameters to get the evidence of a model and compute Bayes factors when comparing two models

The sampling of the posterior distribution is performed using nested sampling with ``ultranest``: https://doi.org/10.21105/joss.03001. The implementation of the likelihoods and priors are implemented in ``src/momenta/stats/model.py``.

Step-by-step usage is described in the following section.

## Use cases




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

* Select the neutrino spectrum and eventually jet model:
```python
import jang.utils.flux
import jang.utils.conversions
flux = jang.utils.flux.FluxFixedPowerLaw(1, 1e6, 2, eref=1)
jet = jang.utils.conversions.JetVonMises(np.deg2rad(10))
pars.set_models(flux, jet=jet)
```
(the list of available jet models is available in ``src/momenta/utils/conversions.py``)

## Detector information
   
* Create/use a YAML file with all relevant information (examples in ``examples/input_files/DETECTORNAME/detector.yaml``)
* Create a new detector object:
```python
from jang.io import NuDetector
det = NuDetector(path_to_yaml_file)
```

* Effective areas have to be defined for each neutrino sample. You can find basic classes in ``src/momenta/io/neutrinos`` and implementation examples in ``examples/``
```python
det.set_effective_areas([effarea1, effarea2, ...])
```

* Any observation can be set with the following commands, where the two arguments are arrays with one entry per sample:
```python
from jang.io.neutrinos import BackgroundFixed
# different background models are available: BackgroundFixed(b0), BackgroundGaussian(b0, deltab), BackgroundPoisson(Noff, alphaoffon)
bkg = [BackgroundFixed(0.51), BackgroundFixed(0.12)]
det.set_observations(n_observed=[0,0], background=bkg)
```

## Source information

* GW database can be easily imported using an existing csv file (see e.g., ``examples/input_files/gw_catalogs/database_example.csv``):
```python
from jang.io import GWDatabase
database_gw = GWDatabase(path_to_csv)
```

* A GW event can be extracted from it:
```python
gw = database_gw.find_gw(name_of_gw, pars)
```

* For point sources, one may use:
```python
from jang.io.transient import PointSource
ps = PointSource(ra_deg=123.45, dec_deg=67.89, name="srcABC")
```

## Obtain results

* Run the nested sampling algorithm:
```python
from jang.stats.run import run_ultranest
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

# Full examples

Some full examples are available in `examples/`:
* `superkamiokande.py` provides a full example using Super-Kamiokande public effective areas from [Zenodo](https://zenodo.org/records/4724823) and expected background rates from [Astrophys.J. 918 (2021) 2, 78](https://doi.org/10.3847/1538-4357/ac0d5a).
* `full_example.ipynb` provides a step-by-step example to get sensitivities and perform a combination of different detectors.