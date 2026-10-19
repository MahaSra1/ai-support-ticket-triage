"""
app.py - the Streamlit UI (display logic only)
Run:  python -m streamlit run app.py
"""
import streamlit as st

from extractor import MODEL, extract_ticket, route_ticket

SAMPLES = {
    "Late parcel": "Hi, I'm Priya. My parcel 33LPX2049881 was meant to arrive last Friday "
                   "and it's still 'in transit'. It's my son's birthday gift, I need it by Saturday!!",
    "Damaged item": "Order arrived today but the box was crushed and the glass vase inside is broken. "
                    "Tracking AP7781234AU. Not happy. - Daniel",
    "Vague message": "hello where is my stuff",
}

st.set_page_config(page_title="AI Support Ticket Triage", page_icon="📦")
st.title("📦 AI Support Ticket Triage")
st.caption(f"Model: {MODEL} via Ollama · structured output · Pydantic validation · rule-based routing")

with st.sidebar:
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
    strict = st.checkbox("Strict schema mode", value=True)
    max_retries = st.number_input("Max retries", 0, 5, 2)

sample = st.selectbox("Sample message", list(SAMPLES))
message = st.text_area("Customer message", SAMPLES[sample], height=120)

if st.button("Create ticket", type="primary"):
    with st.spinner("Reading message..."):
        ticket, attempts, error = extract_ticket(message, temperature, strict, int(max_retries))

    st.caption(f"Attempts used: {len(attempts)}")
    with st.expander("Raw model output(s)"):
        for raw in attempts:
            st.code(raw, language="json")

    if ticket:
        st.success("✅ Valid ticket")
        route = route_ticket(ticket)

        c1, c2, c3 = st.columns(3)
        c1.metric("Issue", ticket.issue_type)
        c2.metric("Urgency", route["urgency"])
        c3.metric("Sentiment", ticket.sentiment)
        st.json(ticket.model_dump())
        st.info(f"Routing: {route['level']} → {route['team']} · respond within {route['sla_hours']}h")
    else:
        st.error(f"❌ Failed after retries: {error}")
