def get_sites(
        datasets,
        event,
        processor,
        structure_array_refs, site_model):
    sites = site_model(
        datasets,
        event,
        processor,
        structure_array_refs)
    return sites
