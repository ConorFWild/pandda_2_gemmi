def rank_events(events, sites, autobuilds, rank_method):

    ranked_event_ids = rank_method(events, sites, autobuilds)

    return ranked_event_ids
