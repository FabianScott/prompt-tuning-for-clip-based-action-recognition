import torch

def module_param_histogram(
    module,
    bins=10,          # int or list/1D tensor of bin edges
    quartiles=False,   # if True, also return Q1, Q2, Q3
    abs_values=True   # if True, use absolute values of parameters
):
    if isinstance(bins, int):
        print(f"Warning, using integer bins may lead to inconsistent binning across calls. Consider providing explicit bin edges.")
    params = torch.cat([p.detach().flatten() for p in module.parameters()])
    if abs_values:
        params = params.abs()

    hist = torch.histogram(
        params,
        bins=bins
    )

    result = {
        "hist": hist.hist,
        "bin_edges": hist.bin_edges,
    }

    if quartiles:
        qs = torch.quantile(params, torch.tensor([0.25, 0.5, 0.75]))
        result["quartiles"] = qs

    return result
