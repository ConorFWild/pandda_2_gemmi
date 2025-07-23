def get_sites(
        datasets,
        event,
ref_dataset,
        # processor,
        # structure_array_refs,
        site_model,
        existing_events,
        existing_sites
):
    sites = site_model(
        datasets,
        event,
        ref_dataset,
        existing_events,
        existing_sites
        # processor,
        # structure_array_refs,
    )
    return sites
