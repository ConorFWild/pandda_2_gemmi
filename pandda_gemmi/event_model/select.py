from ..interfaces import *

def select_model(model_events: Dict[int, Dict[int, EventInterface]], ):
    # max_model_num = max(
    #     model_events,
    #     # key=lambda _event_id: max([((event.build.signal / event.build.noise) * event.local_strength) for event in model_events[_event_id].values()])
    #     key=lambda _event_id: max([event.score for event in model_events[_event_id].values()])
    # )
    max_model_num = max(
        model_events,
        # key=lambda _event_id: max([((event.build.signal / event.build.noise) * event.local_strength) for event in model_events[_event_id].values()])
        key=lambda _event_id: max([event.score for event in model_events[_event_id].values()])
    )
    return max_model_num, model_events[max_model_num]