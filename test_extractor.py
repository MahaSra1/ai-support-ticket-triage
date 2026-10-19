"""
test_extractor.py - tests that run WITHOUT the LLM (fast, free, repeatable)
Run:  python -m pytest -v
"""
import json

import pytest
from pydantic import ValidationError

import extractor
from extractor import compute_urgency, extract_ticket, parse_ticket, route_ticket

GOOD = json.dumps({"customer_name": "Priya", "tracking_number": "33LPX2049881",
                   "issue_type": "late_delivery",
                   "deadline_mentioned": True, "customer_upset": True, "problem_with_order": True,
                   "sentiment": "negative", "summary": "Parcel late, needed by Saturday."})


def make_ticket(deadline=False, upset=False, problem=False, issue="late_delivery"):
    data = json.loads(GOOD)
    data.update(deadline_mentioned=deadline, customer_upset=upset,
                problem_with_order=problem, issue_type=issue)
    return parse_ticket(json.dumps(data))


# ---- validation -------------------------------------------------------------
def test_parse_valid():
    t = parse_ticket(GOOD)
    assert t.issue_type == "late_delivery"
    assert t.deadline_mentioned is True


def test_parse_not_json():
    with pytest.raises(json.JSONDecodeError):
        parse_ticket("Here is your ticket: {...}")


def test_parse_bad_value():
    bad = json.loads(GOOD); bad["issue_type"] = "delayed"      # not an allowed value
    with pytest.raises(ValidationError):
        parse_ticket(json.dumps(bad))


def test_tracking_null_is_allowed():
    data = json.loads(GOOD); data["tracking_number"] = None
    assert parse_ticket(json.dumps(data)).tracking_number is None


def test_tracking_key_missing_is_rejected():
    data = json.loads(GOOD); del data["tracking_number"]
    with pytest.raises(ValidationError):
        parse_ticket(json.dumps(data))


# ---- retry & regex (fake LLM via monkeypatch) --------------------------------
def test_retry_recovers(monkeypatch):
    answers = iter(["not json", GOOD])                        # 1st broken, 2nd good
    monkeypatch.setattr(extractor, "call_llm", lambda *a, **k: next(answers))
    ticket, attempts, error = extract_ticket("my parcel is late")
    assert ticket is not None and error is None
    assert len(attempts) == 2


def test_retry_gives_up(monkeypatch):
    monkeypatch.setattr(extractor, "call_llm", lambda *a, **k: "still not json")
    ticket, attempts, error = extract_ticket("hi", max_retries=2)
    assert ticket is None and len(attempts) == 3


def test_regex_fills_missing_tracking(monkeypatch):
    no_tracking = json.loads(GOOD); no_tracking["tracking_number"] = None
    monkeypatch.setattr(extractor, "call_llm", lambda *a, **k: json.dumps(no_tracking))
    ticket, _, _ = extract_ticket("My parcel 33LPX2049881 is late")
    assert ticket.tracking_number == "33LPX2049881"


def test_regex_does_not_invent_tracking(monkeypatch):
    no_tracking = json.loads(GOOD); no_tracking["tracking_number"] = None
    monkeypatch.setattr(extractor, "call_llm", lambda *a, **k: json.dumps(no_tracking))
    ticket, _, _ = extract_ticket("hello where is my stuff")
    assert ticket.tracking_number is None


# ---- urgency rules ---------------------------------------------------------
def test_urgency_high_when_deadline():
    assert compute_urgency(make_ticket(deadline=True)) == "high"


def test_urgency_high_when_upset():
    assert compute_urgency(make_ticket(upset=True)) == "high"


def test_urgency_medium_when_problem_only():
    assert compute_urgency(make_ticket(problem=True)) == "medium"


def test_urgency_low_for_simple_question():
    assert compute_urgency(make_ticket()) == "low"


# ---- routing rules -----------------------------------------------------------
def test_damaged_is_escalated():
    route = route_ticket(make_ticket(problem=True, issue="damaged"))
    assert route["level"] == "Escalated to team"
    assert route["team"] == "Claims & Damages"


def test_high_urgency_late_is_escalated_with_4h_sla():
    route = route_ticket(make_ticket(deadline=True, problem=True))
    assert route["team"] == "Delivery Operations"
    assert route["sla_hours"] == 4


def test_simple_question_stays_at_l1():
    route = route_ticket(make_ticket(issue="other"))
    assert route["level"] == "L1 support"
    assert route["sla_hours"] == 72
