"""Streamlit chat UI for the IIT Dharwad Academic Office Assistant."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import streamlit as st
from groq import APIStatusError
from rag_chain import FALLBACK_ANSWER, answer_question

LOG_PATH = Path(__file__).resolve().with_name("app.log")
APP_LOGGER = logging.getLogger("iitd_academic_assistant")
APP_LOGGER.setLevel(logging.INFO)
APP_LOGGER.propagate = False
if not any(isinstance(handler, RotatingFileHandler)
           and Path(handler.baseFilename) == LOG_PATH for handler in APP_LOGGER.handlers):
    file_handler = RotatingFileHandler(LOG_PATH, maxBytes=1_000_000,
                                       backupCount=3, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    ))
    APP_LOGGER.addHandler(file_handler)

st.set_page_config(page_title="IIT Dharwad Academic Assistant", page_icon="🎓", layout="centered")
st.title("🎓 IIT Dharwad Academic Office Assistant")
st.caption("Answers are grounded in the academic PDFs indexed in the local document collection.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Conversation")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption("This clears the visible conversation. It does not delete the local document index.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            is_fallback = message["content"].strip() == FALLBACK_ANSWER
            if is_fallback:
                st.info("I retrieved document passages, but could not produce a grounded answer. "
                        "The source list is open so you can check whether the relevant PDF was retrieved.")
            with st.expander(f"Sources ({len(message['sources'])})", expanded=is_fallback):
                for source in message["sources"]:
                    match_note = (f"cosine distance {source['distance']}" if source["distance"] is not None
                                  else "exact term match")
                    st.markdown(f"**{source['source']} — page {source['page']}** · {match_note}")
                    st.caption(source["content"])

if question := st.chat_input("Ask about academic rules, fees, calendars, or programs"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching official documents..."):
            try:
                result = answer_question(question)
                answer = result.answer
                sources = result.sources
            except APIStatusError as exc:
                APP_LOGGER.exception("Groq API request failed (HTTP %s)", exc.status_code)
                if exc.status_code == 413:
                    answer = ("Groq rejected this request because it exceeded the current token limit. "
                              "The assistant already limits prompt size; reduce MAX_CONTEXT_CHUNKS or "
                              "MAX_CONTEXT_CHARS in .env, then restart Streamlit. TOP_K controls how many "
                              "candidates are searched across the full index, not how many are sent to Groq.")
                else:
                    answer = f"Groq API error ({exc.status_code}). Check your model access and API settings."
                sources = []
            except (FileNotFoundError, EnvironmentError, RuntimeError) as exc:
                answer = f"Assistant setup error: {exc}"
                sources = []
            except Exception as exc:  # Keep the chat usable while logging details to the app runtime.
                APP_LOGGER.exception("Assistant request failed")
                st.exception(exc)
                answer = "I could not complete the request because of an internal error. Please check the app logs."
                sources = []
        st.markdown(answer)
        if answer.strip() == FALLBACK_ANSWER:
            APP_LOGGER.warning("Assistant returned fallback; retrieved_source_pages=%s",
                               [f"{item['source']}:{item['page']}" for item in sources])
        if sources:
            is_fallback = answer.strip() == FALLBACK_ANSWER
            if is_fallback:
                st.info("I retrieved document passages, but could not produce a grounded answer. "
                        "The source list is open so you can check whether the relevant PDF was retrieved.")
            with st.expander(f"Sources ({len(sources)})", expanded=is_fallback):
                for source in sources:
                    match_note = (f"cosine distance {source['distance']}" if source["distance"] is not None
                                  else "exact term match")
                    st.markdown(f"**{source['source']} — page {source['page']}** · {match_note}")
                    st.caption(source["content"])
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
