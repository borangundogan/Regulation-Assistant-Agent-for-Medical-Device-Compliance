# src/medcompliance_agent/ui.py

import streamlit as st

from medcompliance_agent.data_loader import load_regulation_text
from medcompliance_agent.chunking import chunk_text
from medcompliance_agent.hybrid_retriever import HybridRetriever
from medcompliance_agent.agent_graph import build_agent_graph

import requests

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="MedCompliance Agent", page_icon="🩺", layout="wide")

st.title("🩺 MedCompliance Agent")
st.write("Generate evidence-bound compliance checklists for medical devices.")

with st.form("device_form"):
    device_name = st.text_input("Device name", "pulse oximeter")
    description = st.text_area("Short description", "wearable sensor for blood oxygen saturation")

    uploaded_pdf = st.file_uploader("Upload regulation PDF (optional)", type=["pdf"])
    submitted = st.form_submit_button("Generate checklist")

if submitted:
    if uploaded_pdf is not None:
        # Use dynamic PDF endpoint
        st.info("Using uploaded PDF for regulations.")
        files = {
            "pdf": (uploaded_pdf.name, uploaded_pdf.getvalue(), "application/pdf")
        }
        data = {
            "device_name": device_name,
            "description": description,
        }

        with st.spinner("Contacting backend API (PDF)..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/generate_from_pdf",
                    files=files,
                    data=data,
                    timeout=120,
                )
                resp.raise_for_status()
                result = resp.json()
            except Exception as e:
                st.error(f"Backend request failed: {e}")
                st.stop()

    else:
        st.info("No PDF uploaded. Using built-in regulation text.")
        payload = {
            "device_name": device_name,
            "description": description,
        }
        with st.spinner("Contacting backend API..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/generate",
                    json=payload,
                    timeout=120,
                )
                resp.raise_for_status()
                result = resp.json()
            except Exception as e:
                st.error(f"Backend request failed: {e}")
                st.stop()

    st.subheader("✅ Checklist")
    st.code(result.get("checklist", ""), language="markdown")

    st.subheader("📊 Evaluation")
    st.json(result.get("evaluation", {}))
