from ..interfaces import *

# def get_event_score(event):
#     return (event.build.signal / event.build.noise) * event.local_strength

def get_event_score(event):
    return event.score

class RankHighEventScoreBySite:
    def __call__(
            self,
            events: Dict[Tuple[str, int], EventInterface],
            sites: Dict[int, SiteInterface],
            autobuilds: Dict[Tuple[str, int], Dict[str, AutobuildInterface]],
    ):
        # # Sort sites ids by best score
        # site_scores = {}
        # for site_id, site in sites.items():
        #     site_scores[site_id] = max(
        #         [
        #             get_event_score(events[_event_id])
        #             for _event_id
        #             in site.event_ids
        #         ]
        #     )
        # # Sort sites ids numerically
        # site_scores = {}
        # for site_id, site in sites.items():
        #     site_scores[site_id] = max(
        #         [
        #             get_event_score(events[_event_id])
        #             for _event_id
        #             in site.event_ids
        #         ]
        #     )

        # Sort event ids within each site by best score
        sorted_event_ids = []
        for _site_id in sorted(sites):
            for event_id in sorted(
                    sites[_site_id].event_ids,
                    key=lambda _event_id: get_event_score(events[_event_id]),
                    reverse=True,
            ):
                sorted_event_ids.append(event_id)

        return sorted_event_ids

