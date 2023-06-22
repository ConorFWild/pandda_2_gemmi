def rank_events(events, autobuilds, rank_method):

    ranked_event_ids = rank_method(events, autobuilds)

    return ranked_event_ids
