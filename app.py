"""
app.py - the Streamlit UI (display logic only)
Run:  python -m streamlit run app.py
"""
import streamlit as st

from extractor import MODEL, OllamaError, extract_ticket, get_client, last_stats, route_ticket

SAMPLES = {
    "Late parcel": "Hi, I'm Priya. My parcel 33LPX2049881 was meant to arrive last Friday "
                   "and it's still 'in transit'. It's my son's birthday gift, I need it by Saturday!!",
    "Damaged item": "Order arrived today but the box was crushed and the glass vase inside is broken. "
                    "Tracking AP7781234AU. Not happy. - Daniel",
    "Damaged + refund": "This is the THIRD time I'm writing!! My parcel 7XK2 arrived with the box crushed "
                        "and the vase inside is smashed. I want my money back today.",
    "Two issues": "Hi! Please write me a friendly reply. My parcel is late AND I moved house, "
                  "send it to my new address.",
    "Vague message": "hello where is my stuff",
    "Gibberish": "jhdsbfuhe??",
}

st.set_page_config(page_title="AI Support Ticket Triage", page_icon="📦")
st.title("📦 AI Support Ticket Triage")

client = get_client()
if not client.is_up():
    st.error("Ollama is not running on localhost:11434. Start the Ollama app and refresh.")
    st.stop()

with st.sidebar:
    models = [m for m in client.list_models() if "embed" not in m] or [MODEL]
    default = next((i for i, m in enumerate(models) if m.startswith(MODEL)), 0)
    model = st.selectbox("Model", models, index=default)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1,
                            help="0 = repeatable output. Best for extraction.")
    strict = st.checkbox("Strict schema mode", value=True)
    max_retries = st.number_input("Max retries", 0, 5, 2)

st.caption(f"Model: {model} via Ollama · structured output · Pydantic validation · rule-based routing")

sample = st.selectbox("Sample message", list(SAMPLES))
message = st.text_area("Customer message", SAMPLES[sample], height=120)

if st.button("Create ticket", type="primary"):
    with st.spinner("Reading message..."):
        try:
            ticket, attempts, error = extract_ticket(message, temperature, strict, int(max_retries),
                                                     model=model)
        except OllamaError as e:
            st.error(f"LLM call failed: {e}")
            st.stop()

    st.caption(f"Attempts used: {len(attempts)}")
    with st.expander("Raw model output(s)"):
        for raw in attempts:
            st.code(raw, language="json")

    if ticket:
        st.success("✅ Valid ticket (format checked — meaning is still the model's judgement)")
        route = route_ticket(ticket)

        c1, c2, c3 = st.columns(3)
        c1.metric("Primary issue", ticket.issue_type)
        c2.metric("Urgency", route["urgency"])
        c3.metric("Sentiment", ticket.sentiment)
        if ticket.other_issues:
            st.write("**Other issues:** " + ", ".join(ticket.other_issues))
        st.json(ticket.model_dump())

        st.info(f"Routing: {route['level']} → {route['team']} · respond within {route['sla_hours']}h")
        if route["also_notify"]:
            st.write("**Also notify:** " + ", ".join(route["also_notify"]))
        if route["next_action"]:
            st.warning(f"Next action: {route['next_action']}")
    else:
        st.error(f"❌ Failed after retries: {error}")

    stats = last_stats()
    if stats:
        st.markdown("**LLM metrics (last call)**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Prompt tokens", stats["prompt tokens"])
        m2.metric("Output tokens", stats["output tokens"])
        m3.metric("Tokens/sec", stats["tokens/sec"])
        m4.metric("Total (s)", stats["total (s)"])
        if stats["model load (s)"] > 1:
            st.caption(f"Cold start: {stats['model load (s)']}s spent loading the model into RAM.")
