def get_sites(
        datasets,
        event,
ref_dataset,
        # processor,
        # structure_array_refs,
        site_model):
    sites = site_model(
        datasets,
        event,
        ref_dataset
        # processor,
        # structure_array_refs,
    )
    return sites
