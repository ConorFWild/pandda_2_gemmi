def filter_selected_events(dtag, selected_events, ):
    selected_model_events = {(dtag, _event_idx): selected_events[_event_idx] for _event_idx in selected_events}
    top_selected_model_events = {
        event_id: selected_model_events[event_id]
        for event_id
        in list(
            sorted(
                selected_model_events,
                # key=lambda _event_id: selected_model_events[_event_id].build.score,
                key=lambda _event_id: selected_model_events[_event_id].build.signal.local_strength * (selected_model_events[_event_id].build.signal / selected_model_events[_event_id].build.noise),
                reverse=True,
            )
        )[:3]
    }

    top_selected_model_events = {(dtag, j+1): event for j, event in enumerate(top_selected_model_events.values())}
    return top_selected_model_events