import ultranest
    
    
def run_mcmc(model):
    sampler = ultranest.ReactiveNestedSampler(model.param_names, model.loglike, model.prior)
    result = sampler.run()
    return {k: v for k, v in zip(model.param_names, result["samples"].transpose())}