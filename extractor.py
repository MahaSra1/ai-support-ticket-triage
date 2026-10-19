"""
extractor.py - the AI logic (no UI here)

Customer message -> structured support ticket -> urgency -> routing.

Design principle: the LLM extracts FACTS, plain code makes DECISIONS.
  - LLM:    name, tracking number, issue type, yes/no urgency signals, sentiment, summary
  - Regex:  fills a missed tracking number
  - Rules:  compute urgency, decide L1 vs escalation, team and SLA
"""
import json
import os
import re
from typing import Literal, Optional

import requests
from pydantic import BaseModel, Field, ValidationError

# ---- 1. CONFIG ---------------------------------------------------------------
# Read from environment variables so nothing is hard-coded.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL = os.getenv("LLM_MODEL", "mistral")


# ---- 2. THE "FORM" (output schema) -------------------------------------------
class Ticket(BaseModel):
    customer_name: Optional[str] = Field(
        ..., description="Customer's name if they mention or sign it, e.g. 'Priya'")
    tracking_number: Optional[str] = Field(
        ..., description="Parcel tracking code exactly as written, e.g. '33LPX2049881' or 'AP7781234AU'")
    issue_type: Literal["late_delivery", "damaged", "lost", "wrong_address", "refund", "other"]
    deadline_mentioned: bool = Field(description="Customer mentions a date or time they need it by")
    customer_upset: bool = Field(description="Customer sounds angry, frustrated or upset")
    problem_with_order: bool = Field(
        description="Something has gone wrong: late, damaged, lost, wrong address")
    sentiment: Literal["positive", "neutral", "negative"]
    summary: str = Field(description="One short sentence")


# ---- 3. THE "JOB DESCRIPTION" (system prompt) --------------------------------
SYSTEM_PROMPT = """You are a customer-support triage assistant for an Australian online store and parcel delivery service.
Read the customer's message and return ONLY a JSON object with these keys:
- customer_name: string or null
- tracking_number: string or null (copy exactly as written)
- issue_type: one of "late_delivery", "damaged", "lost", "wrong_address", "refund", "other"
- deadline_mentioned: true if the customer says when they need it (a day, date or time), else false
- customer_upset: true only if the customer sounds angry, frustrated or upset, else false
- problem_with_order: true if something has gone wrong with the order, else false
- sentiment: one of "positive", "neutral", "negative"
    neutral = plain question with no emotion
- summary: one short sentence
If a value is not in the message, use null. Never invent tracking numbers.
No text before or after the JSON.

Examples:
Message: "hello where is my stuff"
{"customer_name": null, "tracking_number": null, "issue_type": "late_delivery", "deadline_mentioned": false, "customer_upset": false, "problem_with_order": false, "sentiment": "neutral", "summary": "Customer asking where their order is."}

Message: "This is the THIRD time my parcel is late, I need it TOMORROW for a wedding!!"
{"customer_name": null, "tracking_number": null, "issue_type": "late_delivery", "deadline_mentioned": true, "customer_upset": true, "problem_with_order": true, "sentiment": "negative", "summary": "Repeated late delivery, needed tomorrow for a wedding."}"""


# ---- 4. TALK TO THE MODEL ----------------------------------------------------
def call_llm(messages: list, temperature: float = 0.1, strict: bool = True) -> str:
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,                                   # one complete answer
        "format": Ticket.model_json_schema() if strict else "json",
        "options": {"temperature": temperature},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()                                   # fail loudly on HTTP errors
    return r.json()["message"]["content"]


# ---- 5. VALIDATE -------------------------------------------------------------
def parse_ticket(raw: str) -> Ticket:
    data = json.loads(raw)       # step 1: is it JSON?          -> JSONDecodeError
    return Ticket(**data)        # step 2: right keys & values?  -> ValidationError


# ---- 6. REGEX FALLBACK -------------------------------------------------------
# Tracking code: 10-22 uppercase letters/digits, at least one of each
TRACKING_RE = re.compile(r"\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{10,22}\b")


def fill_tracking(ticket: Ticket, message: str) -> Ticket:
    """If the LLM missed the tracking number, find it with regex."""
    if ticket.tracking_number is None:
        m = TRACKING_RE.search(message)
        if m:
            ticket.tracking_number = m.group()
    return ticket


# ---- 7. RULES: URGENCY & ROUTING (no AI) ------------------------------------
def compute_urgency(t: Ticket) -> str:
    """Deterministic, explainable urgency from the extracted signals."""
    if t.deadline_mentioned or t.customer_upset:
        return "high"
    if t.problem_with_order:
        return "medium"
    return "low"


# Example team names - replace with your organisation's real queues.
TEAM_FOR_ISSUE = {
    "late_delivery": "Delivery Operations",
    "lost":          "Investigations",
    "damaged":       "Claims & Damages",
    "wrong_address": "Address & Redirection",
    "refund":        "Customer Payments",
    "other":         "General Enquiries",
}
SLA_HOURS = {"high": 4, "medium": 24, "low": 72}   # target response time


def route_ticket(ticket: Ticket) -> dict:
    """Rules, not AI: decide L1 vs escalation, team and deadline."""
    urgency = compute_urgency(ticket)
    escalate = urgency == "high" or ticket.issue_type in ("damaged", "lost", "refund")
    return {
        "level": "Escalated to team" if escalate else "L1 support",
        "team": TEAM_FOR_ISSUE[ticket.issue_type] if escalate else "L1 Customer Support",
        "sla_hours": SLA_HOURS[urgency],
        "urgency": urgency,
    }


# ---- 8. FULL PIPELINE WITH RETRY ---------------------------------------------
def extract_ticket(message: str, temperature: float = 0.1, strict: bool = True, max_retries: int = 2):
    """Returns (ticket or None, list_of_raw_attempts, last_error or None)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message},
    ]
    attempts, last_error = [], None
    for _ in range(max_retries + 1):
        raw = call_llm(messages, temperature, strict)
        attempts.append(raw)
        try:
            return fill_tracking(parse_ticket(raw), message), attempts, None
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user",
                             "content": f"Your JSON was invalid: {e}. Return the corrected JSON only."})
    return None, attempts, last_error
