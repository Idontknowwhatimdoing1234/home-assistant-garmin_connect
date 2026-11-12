"""Tests for merging Garmin Coach plan detail into scheduled sessions.

Data shapes follow a real Garmin Run Coach plan (FBT_ADAPTIVE), trimmed to
the fields that matter: two sessions on the same day that share one
calendar id, durations only in the plan's taskList.
"""

from datetime import date
from typing import Any
from unittest.mock import AsyncMock

import pytest
from ha_garmin.exceptions import GarminAPIError, GarminAuthError
from pytest_homeassistant.common import MockConfigEntry

from custom_components.garmin_connect.calendar import _event_from_workout, _with_steps
from custom_components.garmin_connect.coach_plan import (
    enrich_detailed,
    merge_coach_plan,
    plan_id_from_items,
)

TODAY = date(2026, 11, 12)
PLAN_ID = 48562098
SHARED_ID = 1794443561000  # both of today's sessions carry this id


def _item(item_id: int, day: str, title: str, sport: str, uuid: str) -> dict[str, Any]:
    """A raw calendarItems entry of a Coach session (subset of ~70 fields)."""
    return {
        "id": item_id,
        "itemType": "fbtAdaptiveWorkout",
        "trainingPlanId": PLAN_ID,
        "title": title,
        "date": day,
        "duration": None,
        "sportTypeKey": sport,
        "atpPlanId": None,
        "workoutId": None,
        "workoutUuid": uuid,
        "tpType": "FBT_ADAPTIVE",
    }


def _task(day: str, order: int, name: str, sport: str, secs: int, uuid: str, desc=None):
    """A taskList entry of the adaptive plan detail."""
    return {
        "trainingPlanId": PLAN_ID,
        "workoutOrder": order,
        "calendarDate": day,
        "keyWorkout": False,
        "taskWorkout": {
            "workoutId": None,
            "sportType": {"sportTypeKey": sport},
            "workoutName": name,
            "workoutDescription": desc,
            "estimatedDurationInSecs": secs,
            "workoutUuid": uuid,
            "trainingEffectLabel": "AEROBIC_BASE",
            "restDay": False,
        },
    }


CALENDAR_ITEMS = [
    _item(SHARED_ID, "2026-11-12", "Aktivierung vor dem Lauf 3", "strength_training", "u-act"),
    _item(SHARED_ID, "2026-11-12", "Basis", "running", "u-base"),
    _item(1794529961000, "2026-11-13", "Basis", "running", "u-fri"),
    # yesterday: dropped
    _item(1794357161000, "2026-11-11", "Basis", "running", "u-old"),
    # a weigh-in: not a session
    {"id": 1, "itemType": "weight", "date": "2026-11-12"},
]

PLAN = {
    "trainingPlanId": PLAN_ID,
    "taskList": [
        _task("2026-11-12", 1, "Aktivierung vor dem Lauf 3", "strength_training", 1200, "u-act"),
        _task("2026-11-12", 2, "Basis", "running", 3300, "u-base", "151bpm"),
        _task("2026-11-13", 1, "Basis", "running", 2220, "u-fri", "151bpm"),
        # not on the calendar yet
        _task("2026-11-15", 1, "Langer Lauf", "running", 4140, "u-sun", "151bpm"),
    ],
}

# What ha-garmin 0.2.1 makes of it: only one of the two same-id sessions.
LIBRARY_SCHEDULED = [
    {"id": SHARED_ID, "date": "2026-11-12", "title": "Basis", "sportTypeKey": "running"},
    {"id": 1794529961000, "date": "2026-11-13", "title": "Basis", "sportTypeKey": "running"},
]


def test_merge_keeps_both_same_id_sessions_with_duration() -> None:
    sessions = merge_coach_plan(LIBRARY_SCHEDULED, CALENDAR_ITEMS, PLAN, TODAY)

    today = [s for s in sessions if s["date"] == "2026-11-12"]
    assert [s["title"] for s in today] == ["Aktivierung vor dem Lauf 3", "Basis"]
    assert [s["estimatedDurationInSecs"] for s in today] == [1200, 3300]
    assert today[1]["workoutDescription"] == "151bpm"
    assert "workoutDescription" not in today[0]  # None is left out


def test_merge_drops_past_and_non_session_items() -> None:
    sessions = merge_coach_plan([], CALENDAR_ITEMS, PLAN, TODAY)
    assert all(s["date"] >= "2026-11-12" for s in sessions)
    assert "u-old" not in {s["workoutUuid"] for s in sessions}


def test_merge_adds_plan_sessions_not_on_calendar() -> None:
    sessions = merge_coach_plan([], CALENDAR_ITEMS, PLAN, TODAY)
    sunday = [s for s in sessions if s["date"] == "2026-11-15"]
    assert sunday == [
        {
            "id": None,
            "date": "2026-11-15",
            "title": "Langer Lauf",
            "sportTypeKey": "running",
            "trainingPlanId": PLAN_ID,
            "workoutUuid": "u-sun",
            "estimatedDurationInSecs": 4140,
            "workoutDescription": "151bpm",
            "trainingEffectLabel": "AEROBIC_BASE",
            "keyWorkout": False,
            "workoutOrder": 1,
        }
    ]


def test_merge_skips_rest_days() -> None:
    rest = _task("2026-11-16", 1, "Ruhetag", "running", 0, "u-rest")
    rest["taskWorkout"]["restDay"] = True
    sessions = merge_coach_plan([], [], {"taskList": [rest]}, TODAY)
    assert sessions == []


def test_merge_deduplicates_month_boundary_copies() -> None:
    """Both month views return boundary days; each session stays once."""
    sessions = merge_coach_plan([], CALENDAR_ITEMS + CALENDAR_ITEMS, {}, TODAY)
    assert len(sessions) == 3


def test_merge_without_calendar_items_uses_library_list() -> None:
    sessions = merge_coach_plan(LIBRARY_SCHEDULED, [], {}, TODAY)
    assert sessions == LIBRARY_SCHEDULED
    assert sessions[0] is not LIBRARY_SCHEDULED[0]  # copies, not the originals


def test_plan_id_from_items() -> None:
    assert plan_id_from_items(CALENDAR_ITEMS, []) == PLAN_ID
    # 0 and bools are not plan ids; atpPlanId is the fallback
    assert plan_id_from_items([{"trainingPlanId": 0, "atpPlanId": 7}], []) == 7
    assert plan_id_from_items([{"trainingPlanId": True}], []) is None
    assert plan_id_from_items([], [{"atpPlanId": 9}]) == 9


def test_enrich_detailed_matches_id_and_title() -> None:
    sessions = merge_coach_plan([], CALENDAR_ITEMS, PLAN, TODAY)
    detailed = {"id": SHARED_ID, "title": "Basis", "workoutSteps": [{"x": 1}]}
    enriched = enrich_detailed(detailed, sessions)
    assert enriched["estimatedDurationInSecs"] == 3300
    assert enriched["workoutSteps"] == [{"x": 1}]  # library fields kept
    assert enrich_detailed({}, sessions) == {}


def test_calendar_event_shows_duration_and_target() -> None:
    sessions = merge_coach_plan([], CALENDAR_ITEMS, PLAN, TODAY)
    events = [_event_from_workout(s) for s in sessions if s["date"] == "2026-11-12"]
    assert [e.summary for e in events] == ["Aktivierung vor dem Lauf 3", "Basis"]
    assert events[0].description == "strength_training\n20 min"
    assert events[1].description == "running\n55 min\n151bpm"
    # same calendar id, distinct uids
    assert [e.uid for e in events] == ["u-act", "u-base"]


def test_with_steps_does_not_mix_same_id_sessions() -> None:
    """The library's detailed today workout must not replace its id twin."""
    sessions = merge_coach_plan([], CALENDAR_ITEMS, PLAN, TODAY)
    data = {"todayScheduledWorkout": {"id": SHARED_ID, "title": "Basis"}}
    activation, base = sessions[0], sessions[1]
    assert _with_steps(data, activation)["title"] == "Aktivierung vor dem Lauf 3"
    assert _with_steps(data, base)["estimatedDurationInSecs"] == 3300


# ── ActivityCoordinator integration ──────────────────────────────────────────


async def _activity_refresh(entry: MockConfigEntry) -> dict[str, Any]:
    coordinator = entry.runtime_data.activity
    await coordinator.async_refresh()
    return coordinator.data


@pytest.fixture
def coach_client(mock_client):
    """mock_client answering with the Coach calendar and plan above."""
    mock_client.get_scheduled_workouts = AsyncMock(
        side_effect=[{"calendarItems": CALENDAR_ITEMS}, {"calendarItems": []}] * 10
    )
    mock_client.get_adaptive_training_plan_by_id = AsyncMock(return_value=PLAN)
    return mock_client


async def test_coordinator_merges_coach_plan(
    init_integration: MockConfigEntry, coach_client
) -> None:
    coach_client.today.return_value = TODAY
    data = await _activity_refresh(init_integration)

    titles = [s["title"] for s in data["todayScheduledWorkouts"]]
    assert titles == ["Aktivierung vor dem Lauf 3", "Basis"]
    assert data["scheduledWorkouts"][0]["estimatedDurationInSecs"] == 1200
    # current and next month, like the library
    coach_client.get_scheduled_workouts.assert_any_await(2026, 11)
    coach_client.get_scheduled_workouts.assert_any_await(2026, 12)
    coach_client.get_adaptive_training_plan_by_id.assert_awaited_with(PLAN_ID)


async def test_coordinator_caches_plan(init_integration: MockConfigEntry, coach_client) -> None:
    coach_client.today.return_value = TODAY
    await _activity_refresh(init_integration)
    await _activity_refresh(init_integration)
    assert coach_client.get_adaptive_training_plan_by_id.await_count == 1


async def test_coordinator_keeps_library_data_on_api_error(
    init_integration: MockConfigEntry, mock_client
) -> None:
    mock_client.get_scheduled_workouts = AsyncMock(side_effect=GarminAPIError("boom"))
    mock_client.get_adaptive_training_plan_by_id = AsyncMock(side_effect=GarminAPIError("x"))
    data = await _activity_refresh(init_integration)
    assert data["scheduledWorkouts"][0]["title"] == "Benchmark Run"
    assert "todayScheduledWorkouts" not in data


async def test_coordinator_auth_error_propagates(
    init_integration: MockConfigEntry, mock_client
) -> None:
    mock_client.get_scheduled_workouts = AsyncMock(side_effect=GarminAuthError("expired"))
    coordinator = init_integration.runtime_data.activity
    await coordinator.async_refresh()
    assert coordinator.last_update_success is False
