from agent.scheduler.event import Event, score_event


def test_event_id_uniqueness():
    """Test that each event gets a unique ID."""

    event1 = Event(
        name="Event 1",
        cost=10.0,
        duration=2,
        base_exp=100.0,
        opening_hours=None
    )

    event2 = Event(
        name="Event 1",
        cost=10.0,
        duration=2,
        base_exp=100.0,
        opening_hours=None
    )
    # default id is generated from uuid; so the two events should have different ids
    assert event1.id != event2.id


def test_score_event():
    event1 = Event(
        name="Event 1",
        cost=25.0,
        duration=2,
        base_exp=100.0,
        opening_hours=None
    )
    score = score_event(event1, actual_duration=1, w_xp=1.0,
                        w_count=1.0, w_cost=1.0, w_dur=1.0)
    expected_score = 100.0 / 2 * 1.0 + 1.0 * 1.0 - 25.0 * 1.0 - 1 * 1.0
    assert score == expected_score
