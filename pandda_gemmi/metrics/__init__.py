import numpy as np


def _get_event_probability(_event_score, event_score_quantiles):
    p = np.interp(_event_score, event_score_quantiles['x'], event_score_quantiles['p'])
    return p


def get_hit_in_site_probabilities(pandda_events, ranking, sites, event_score_quantiles):
    hit_in_site_probabilities = {}
    for _site_id, site in sites.items():
        # Get site events
        site_events = site.event_ids

        # For each event, get site events with lower ranks, then get binomial probability hits != 0
        for event_id in site_events:
            event_ranking = ranking.index(event_id)
            lower_ranked_events = [_event_id for _event_id in site_events if ranking.index(_event_id) > event_ranking]
            event_scores = [pandda_events[_event_id].score for _event_id in lower_ranked_events]
            event_probabilities = [_get_event_probability(_event_score, event_score_quantiles) for _event_score in
                                   event_scores]
            if len(event_probabilities) != 0:
                hit_in_site_probabilities[event_id] = np.prod(
                    [1 - _event_probability for _event_probability in event_probabilities])
            else:
                hit_in_site_probabilities[event_id] = 0.0

    return hit_in_site_probabilities
