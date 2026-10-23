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
                   "issues": ["late_delivery"],
                   "deadline_mentioned": True, "customer_upset": True, "problem_with_order": True,
                   "sentiment": "negative", "summary": "Parcel late, needed by Saturday."})


def make_ticket(deadline=False, upset=False, problem=False, issue="late_delivery"):
    data = json.loads(GOOD)
    issues = issue if isinstance(issue, list) else [issue]
    data.update(deadline_mentioned=deadline, customer_upset=upset,
                problem_with_order=problem, issues=issues)
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
    bad = json.loads(GOOD); bad["issues"] = ["delayed"]        # not an allowed value
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


# ---- multi-issue + primary-issue rules ---------------------------------------
def test_empty_issues_rejected():
    data = json.loads(GOOD); data["issues"] = []
    with pytest.raises(ValidationError):
        parse_ticket(json.dumps(data))


def test_cause_beats_request():
    """'Box crushed, I want my money back' -> primary is damaged, not refund."""
    t = make_ticket(issue=["refund", "damaged"])
    assert t.issue_type == "damaged"
    assert t.other_issues == ["refund"]


def test_duplicates_removed_and_unclear_dropped_when_specific():
    t = make_ticket(issue=["late_delivery", "unclear", "late_delivery"])
    assert t.issues == ["late_delivery"]


def test_issue_type_in_dump():
    assert make_ticket(issue=["wrong_address", "late_delivery"]).model_dump()["issue_type"] == "wrong_address"


def test_two_issues_notify_second_team():
    route = route_ticket(make_ticket(problem=True, issue=["late_delivery", "wrong_address"]))
    assert route["team"] == "L1 Customer Support"          # medium, not an escalation issue
    assert route["also_notify"] == ["Delivery Operations"]


def test_damaged_refund_escalates_and_notifies_payments():
    route = route_ticket(make_ticket(upset=True, problem=True, issue=["damaged", "refund"]))
    assert route["team"] == "Claims & Damages"
    assert route["also_notify"] == ["Customer Payments"]


def test_unclear_stays_l1_and_asks_for_details():
    route = route_ticket(make_ticket(issue="unclear"))
    assert route["level"] == "L1 support"
    assert route["urgency"] == "low"
    assert route["next_action"] == "Ask the customer for more details"


def test_schema_sent_to_llm_has_no_computed_field():
    """The LLM fills 'issues'; issue_type is computed by code, so it must not be in the schema."""
    props = extractor.Ticket.model_json_schema()["properties"]
    assert "issues" in props and "issue_type" not in props
