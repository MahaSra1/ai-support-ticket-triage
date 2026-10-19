# 📦 AI Support Ticket Triage

![tests](https://github.com/MahaSra1/ai-support-ticket-triage/actions/workflows/tests.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![LLM](https://img.shields.io/badge/LLM-Mistral%207B%20(Ollama)-orange)
![License](https://img.shields.io/badge/license-MIT-green)

Turns messy customer messages about parcels and online orders into **structured support tickets**, then **routes** them to L1 support or the right team with a response-time target (SLA). It mirrors a real parcel-support workflow: *create ticket → L1 → escalate to a team*.

> "Hi, I'm Priya. My parcel 33LPX2049881 was meant to arrive last Friday… I need it by Saturday!!"

```json
{
  "customer_name": "Priya",
  "tracking_number": "33LPX2049881",
  "issue_type": "late_delivery",
  "deadline_mentioned": true,
  "customer_upset": true,
  "problem_with_order": true,
  "sentiment": "negative",
  "summary": "Parcel late, needed by Saturday for a birthday gift."
}
```
**→ Urgency: high · Escalated to Delivery Operations · respond within 4h**

---

## ✨ Features

- **LLM extraction** with a local model (Mistral 7B via Ollama) — no paid API, data stays on the machine
- **Structured output**: JSON-schema mode (constrained decoding) generated from a Pydantic model
- **Two-step validation** (`json.loads` → Pydantic) with an **automatic retry loop** that feeds the error back to the model
- **Regex fallback** for tracking numbers the model misses
- **Deterministic urgency and routing rules** — the LLM extracts facts, code makes decisions
- **16 unit tests** that run without the LLM (mocked with `monkeypatch`), plus GitHub Actions CI
- Clean separation: `extractor.py` (logic) · `app.py` (UI) · `test_extractor.py` (tests)

## 🖼️ Screenshots

| "Need it by today" → high, escalated | "Where is my stuff" → medium, L1 |
|---|---|
| ![Deadline escalated](docs/screenshots/deadline_escalated.png) | ![Vague message](docs/screenshots/vague_message.png) |

## 🏗️ Architecture

```mermaid
flowchart TD
    A[Customer message] --> B[Build messages<br/>system rules + few-shot examples]
    B --> C[call_llm → Ollama / Mistral<br/>JSON-schema mode]
    C --> D{json.loads OK?}
    D -- no --> R[Retry: show model its answer + error<br/>max 2 retries]
    D -- yes --> E{Pydantic Ticket valid?}
    E -- no --> R
    R --> C
    E -- yes --> F[Regex fallback<br/>fill missing tracking number]
    F --> G[compute_urgency<br/>rules]
    G --> H[route_ticket<br/>L1 or team + SLA]
    H --> I[Streamlit UI]
```

### Who decides what

| Job | Done by | Why |
|---|---|---|
| Understand the message (issue, signals, tone, summary) | **LLM** | Needs language understanding |
| Tracking number (if missed) | **Regex** | Fixed pattern — reliable, fast, free |
| Format + allowed values | **Pydantic** | Strict, typed validation |
| Urgency | **Rules** | Stable, explainable, testable |
| Routing + SLA | **Rules** | Must be predictable and auditable |

### Urgency and routing rules

| Signals from the LLM | Urgency | SLA |
|---|---|---|
| `deadline_mentioned` or `customer_upset` | high | 4h |
| only `problem_with_order` | medium | 24h |
| none | low | 72h |

Escalate to a team if urgency is **high** or the issue is **damaged / lost / refund**; otherwise **L1 Customer Support**.

| Issue type | Team (when escalated) |
|---|---|
| late_delivery | Delivery Operations |
| lost | Investigations |
| damaged | Claims & Damages |
| wrong_address | Address & Redirection |
| refund | Customer Payments |
| other | General Enquiries |

*Team names are examples and can be changed in `TEAM_FOR_ISSUE`.*

## 🚀 Getting started

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.com) installed and running.

```bash
git clone https://github.com/MahaSra1/ai-support-ticket-triage.git
cd ai-support-ticket-triage

python -m venv venv
# Windows:  venv\Scripts\activate
# macOS/Linux:  source venv/bin/activate

pip install -r requirements.txt
ollama pull mistral

python -m streamlit run app.py          # opens http://localhost:8501
```

Optional — use a different model:
```bash
# PowerShell
$env:LLM_MODEL="qwen2.5:7b"
# macOS/Linux
export LLM_MODEL=qwen2.5:7b
```

## 🧪 Tests

```bash
python -m pytest -v
```

The LLM call is replaced with a fake (`monkeypatch`), so tests run in under a second, cost nothing and give the same result every time.

| Area | What is tested |
|---|---|
| Validation | valid JSON, non-JSON, disallowed values, null vs missing keys |
| Retry | recovers after a bad answer; gives up after max retries |
| Regex | fills a missed tracking number; never invents one |
| Urgency | high (deadline / upset), medium (problem only), low |
| Routing | damaged → Claims & Damages; high → 4h SLA; simple question → L1 |

## 🔧 Challenges I solved

| Problem found during testing | Fix | Technique |
|---|---|---|
| Schema mode returned **valid JSON with empty name/tracking** (the tracking number even appeared in the summary) | Required fields with descriptions in the schema | Schema design |
| Tracking number still null in strict mode | Deterministic **regex fallback** | LLM + plain code |
| Making fields required broke an existing test | Updated tests to document the new contract | Contract testing |
| Vague messages rated **high urgency** | Few-shot examples | Prompt engineering |
| Urgency **changed when the same question was reworded** | Redesign: LLM extracts yes/no facts, **rules compute urgency** | Facts vs decisions |
| `NameError` in the UI after adding routing | Moved import to the top of the file | Debugging / PEP 8 |

**Key lesson:** valid output is not the same as correct output. Validation catches format errors; meaning errors need clear prompts, examples, deterministic rules and evaluation.

## ⚠️ Known limitations

- The tracking-number regex is generic and can match words like `SATURDAY2024`; production would use each carrier's exact format.
- Sentiment and `customer_upset` are still LLM judgements and can be inconsistent on short messages.
- No evaluation set yet — accuracy is checked manually on sample messages.
- Single message at a time; no batch processing.

## 🏢 How this would run in production

| This project | Production |
|---|---|
| Ollama + Mistral on a laptop | Azure OpenAI / AWS Bedrock in an Australian region, or self-hosted vLLM |
| Streamlit button | FastAPI endpoint behind a queue, called by the CRM / email system |
| Prompt in a Python string | Versioned prompts, A/B tested |
| Unit tests | Unit tests + labelled evaluation set measuring field accuracy |
| Names in prompts | PII masking and data-retention rules |
| Retry loop | Retry + fallback model + human-review queue |

## 🗺️ Roadmap

- [ ] Evaluation set (50+ labelled messages) with field-level accuracy
- [ ] CSV batch upload → ticket table → download
- [ ] FastAPI endpoint (`POST /tickets`)
- [ ] L1 auto-answers with a RAG FAQ chatbot (delivery and returns policies)
- [ ] Agent that looks up tracking status with tool calling
- [ ] Docker + cloud deployment

## 🧰 Tech stack

Python · Ollama · Mistral 7B · Pydantic v2 · Streamlit · Requests · Regex · Pytest · GitHub Actions

## 👩‍💻 Author

**Mahalaxmi (Maha)** — Python engineer moving from automotive ADAS validation into AI engineering.
GitHub: [@MahaSra1](https://github.com/MahaSra1)

## 📄 License

MIT — see [LICENSE](LICENSE).
