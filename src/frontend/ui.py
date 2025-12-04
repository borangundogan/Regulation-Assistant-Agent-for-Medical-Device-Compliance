# src/medcompliance_agent/ui.py

import streamlit as st

from medcompliance_agent.data_loader import load_regulation_text
from medcompliance_agent.chunking import chunk_text
from medcompliance_agent.hybrid_retriever import HybridRetriever
from medcompliance_agent.agent_graph import build_agent_graph

import requests

st.set_page_config(page_title="MedCompliance Agent", page_icon="🩺", layout="wide")

@st.cache_resource
def load_agent():
    text = load_regulation_text()
    chunks = chunk_text(text)
    retriever = HybridRetriever(chunks)
    run_agent = build_agent_graph(retriever)
    return run_agent

run_agent = load_agent()

st.title("🩺 MedCompliance Agent")
st.write("Generate evidence-bound compliance checklists for medical devices.")

with st.form("device_form"):
    device_name = st.text_input("Device name", "pulse oximeter")
    description = st.text_area("Short description", "wearable sensor for blood oxygen saturation")
    submitted = st.form_submit_button("Generate checklist")

if submitted:
    payload = {
        "device_name": device_name,
        "description": description
    }

    with st.spinner("Contacting backend API..."):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/generate",
                json=payload,
                timeout=120
            )
            data = response.json()
        except Exception as e:
            st.error(f"Backend request failed: {e}")
            st.stop()

    st.subheader("✅ Checklist")
    st.code(data.get("checklist", ""), language="markdown")

    st.subheader("📊 Evaluation")
    st.text(data.get("evaluation", {}))

    st.subheader("🧠 Raw JSON")
    st.json(data)
