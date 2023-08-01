def filter_selected_events(dtag, selected_events, ):
    selected_model_events = {(dtag, _event_idx): selected_events[_event_idx] for _event_idx in selected_events}
    top_selected_model_events = {
        event_id: selected_model_events[event_id]
        for event_id
        in list(
            sorted(
                selected_model_events,
                key=lambda _event_id: selected_model_events[_event_id].score,
                reverse=True,
            )
        )[:3]
    }