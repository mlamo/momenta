import logging
import ultranest


logger = logging.getLogger("ultranest")
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.WARNING)

    
def run_mcmc(model):
    sampler = ultranest.ReactiveNestedSampler(model.param_names, model.loglike, model.prior)
    result = sampler.run(show_status=False, viz_callback=False)
    return {k: v for k, v in zip(model.param_names, result["samples"].transpose())}
