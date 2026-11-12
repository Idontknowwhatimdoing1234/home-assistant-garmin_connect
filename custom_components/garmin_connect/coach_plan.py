"""Merge Garmin Coach (adaptive plan) detail into the scheduled sessions.

ha-garmin's fetch_activity_data builds ``scheduledWorkouts`` from the
training calendar alone. For Garmin Coach / adaptive plans that loses two
things:

* Duration and target. Calendar items of Coach sessions carry no
  ``workoutId`` and ``duration`` is null, so there is nothing to look up.
  The plan's own detail (``get_adaptive_training_plan_by_id``) has a
  ``taskList`` whose entries carry ``estimatedDurationInSecs`` and a short
  ``workoutDescription`` ("151bpm", "15:00@4:50/km").
* Sessions sharing an id. Two Coach sessions on the same day (e.g. a
  pre-run activation plus the run) can come back with the same calendar
  ``id``; the library deduplicates on (id, itemType, date) and keeps only
  one of them.

Calendar item and plan task are matched on ``workoutUuid``, which both
carry (the library trims it from its own output, so the raw calendar items
are needed). Observed against a real Coach plan (Garmin Run Coach,
FBT_ADAPTIVE), November 2026.

Everything here is pure: no I/O, so it is tested without Home Assistant.
"""

from __future__ import annotations

from datetime import date
from typing import Any

# Calendar item types that are scheduled sessions (same as ha-garmin's
# CALENDAR_WORKOUT_ITEM_TYPES).
WORKOUT_ITEM_TYPES = ("workout", "fbtAdaptiveWorkout")

# Fields kept from a raw calendar item: ha-garmin's trimmed set plus
# workoutUuid, the key to the plan task.
CALENDAR_KEYS = (
    "id",
    "date",
    "title",
    "sportTypeKey",
    "workoutId",
    "atpPlanId",
    "trainingPlanId",
    "protectedWorkoutSchedule",
    "phasedTrainingPlan",
    "duration",
    "distance",
    "calories",
    "workoutUuid",
)

# Fields copied from a plan task's taskWorkout / task onto the session.
PLAN_WORKOUT_KEYS = (
    "estimatedDurationInSecs",
    "workoutDescription",
    "trainingEffectLabel",
)
PLAN_TASK_KEYS = ("keyWorkout", "workoutOrder")


def plan_id_from_items(
    calendar_items: list[dict[str, Any]], scheduled: list[dict[str, Any]]
) -> int | None:
    """Return the adaptive plan id the sessions belong to, if any.

    trainingPlanId on the calendar items is the plan id (48562098 in the
    observed plan). It has been seen as 0 on other item kinds, so only
    positive ints count; atpPlanId is the fallback.
    """
    for item in [*calendar_items, *scheduled]:
        for key in ("trainingPlanId", "atpPlanId"):
            value = item.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
    return None


def _session_key(item: dict[str, Any]) -> tuple[Any, ...]:
    """Identity of one session: the id alone is not unique (see module doc)."""
    return (item.get("id"), item.get("itemType"), item.get("date"), item.get("workoutUuid"))


def _calendar_sessions(calendar_items: list[dict[str, Any]], today: date) -> list[dict[str, Any]]:
    """Scheduled sessions from today on, trimmed, without dropping same-id pairs."""
    today_str = today.isoformat()
    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    for item in calendar_items:
        if item.get("itemType") not in WORKOUT_ITEM_TYPES:
            continue
        if (item.get("date") or "") < today_str:
            continue
        # Both month views cover the boundary days, so the same item can
        # arrive twice; setdefault keeps the first copy.
        seen.setdefault(_session_key(item), {k: item.get(k) for k in CALENDAR_KEYS})
    return list(seen.values())


def _plan_fields(task: dict[str, Any]) -> dict[str, Any]:
    """The fields a plan task adds to a session (None values left out)."""
    workout = task.get("taskWorkout") or {}
    fields = {k: workout.get(k) for k in PLAN_WORKOUT_KEYS}
    fields.update({k: task.get(k) for k in PLAN_TASK_KEYS})
    return {k: v for k, v in fields.items() if v is not None}


def _session_from_task(task: dict[str, Any], plan_id: Any) -> dict[str, Any] | None:
    """A session for a plan task Garmin has not put on the calendar yet.

    Garmin commits a Coach session to the calendar only shortly before it
    happens; the plan already lists the rest of the week. These get no
    calendar id.
    """
    workout = task.get("taskWorkout") or {}
    if workout.get("restDay") or not task.get("calendarDate"):
        return None
    return {
        "id": None,
        "date": task["calendarDate"],
        "title": workout.get("workoutName"),
        "sportTypeKey": (workout.get("sportType") or {}).get("sportTypeKey"),
        "trainingPlanId": task.get("trainingPlanId", plan_id),
        "workoutUuid": workout.get("workoutUuid"),
        **_plan_fields(task),
    }


def merge_coach_plan(
    scheduled: list[dict[str, Any]],
    calendar_items: list[dict[str, Any]],
    plan: dict[str, Any],
    today: date,
) -> list[dict[str, Any]]:
    """Return the scheduled sessions with Coach plan detail merged in.

    scheduled:      ha-garmin's scheduledWorkouts (used if the raw calendar
                    items could not be fetched)
    calendar_items: raw calendarItems of the current and next month
    plan:           adaptive plan detail, {} if not available
    today:          Garmin calendar date; earlier sessions are dropped

    Sorted by date, then by the plan's order within the day.
    """
    sessions = (
        _calendar_sessions(calendar_items, today)
        if calendar_items
        else [dict(item) for item in scheduled]
    )

    today_str = today.isoformat()
    tasks = [
        task for task in plan.get("taskList") or [] if (task.get("calendarDate") or "") >= today_str
    ]
    # workoutUuid -> plan task; tasks without a uuid cannot be matched.
    by_uuid = {
        (task.get("taskWorkout") or {}).get("workoutUuid"): task
        for task in tasks
        if (task.get("taskWorkout") or {}).get("workoutUuid")
    }

    matched: set[Any] = set()
    for session in sessions:
        uuid = session.get("workoutUuid")
        task = by_uuid.get(uuid) if uuid else None
        if task is not None:
            session.update(_plan_fields(task))
            matched.add(uuid)

    # Plan sessions not (yet) on the calendar.
    plan_id = plan.get("trainingPlanId")
    for uuid, task in by_uuid.items():
        if uuid in matched:
            continue
        new_session = _session_from_task(task, plan_id)
        if new_session is not None:
            sessions.append(new_session)

    return sorted(
        sessions,
        key=lambda s: (s.get("date") or "", s.get("workoutOrder") or 0),
    )


def enrich_detailed(detailed: dict[str, Any], sessions: list[dict[str, Any]]) -> dict[str, Any]:
    """Copy plan fields onto ha-garmin's today/next workout (which has steps).

    That dict carries no workoutUuid, so it is matched on id and title --
    the id alone can belong to two sessions.
    """
    if not detailed:
        return detailed
    for session in sessions:
        if session.get("id") == detailed.get("id") and session.get("title") == detailed.get(
            "title"
        ):
            extra = {k: v for k, v in session.items() if k not in detailed or detailed[k] is None}
            return {**detailed, **extra}
    return detailed
