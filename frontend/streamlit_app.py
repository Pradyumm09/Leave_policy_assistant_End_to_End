import os

import requests
import streamlit as st


# ============================================================
# Configuration
# ============================================================

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://127.0.0.1:8000",
)


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Leave Policy Assistant",
    page_icon="📋",
    layout="centered",
)


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .source-box {
        padding: 0.9rem;
        border-radius: 0.5rem;
        background-color: #262730;
        color: #ffffff;
        border-left: 4px solid #6366f1;
        margin-bottom: 0.6rem;
        line-height: 1.6;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Header
# ============================================================

st.markdown(
    '<div class="main-title">📋 Leave Policy Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Ask questions about the organization's leave policy."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# Session State
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("⚙️ Assistant")

    st.caption(
        "Agentic RAG Leave Policy Assistant"
    )

    st.divider()

    st.subheader("Backend")

    st.code(
        BACKEND_URL,
        language="text",
    )

    # --------------------------------------------------------
    # Health Check
    # --------------------------------------------------------

    try:

        response = requests.get(
            f"{BACKEND_URL}/health",
            timeout=5,
        )

        if response.status_code == 200:

            st.success(
                "Backend Connected"
            )

        else:

            st.error(
                "Backend Unavailable"
            )

    except requests.RequestException:

        st.error(
            "Backend Unavailable"
        )

    st.divider()

    # --------------------------------------------------------
    # Clear Conversation
    # --------------------------------------------------------

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.caption(
        "Answers are grounded in the indexed "
        "organization leave policy."
    )


# ============================================================
# Display Previous Messages
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        # ----------------------------------------------------
        # Previous Sources
        # ----------------------------------------------------

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                "📚 Sources"
            ):

                for source in message[
                    "sources"
                ]:

                    source_id = source.get(
                        "id"
                    ) or "Unknown"

                    section = source.get(
                        "section"
                    ) or "Unknown"

                    source_type = source.get(
                        "type"
                    ) or "Unknown"

                    st.markdown(
                        f"""
                        <div class="source-box">
                        <b>Document:</b> {source_id}<br>
                        <b>Section:</b> {section}<br>
                        <b>Type:</b> {source_type}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# ============================================================
# Chat Input
# ============================================================

question = st.chat_input(
    "Ask a leave policy question..."
)


# ============================================================
# Process Question
# ============================================================

if question:

    question = question.strip()

    if not question:

        st.warning(
            "Please enter a question."
        )

        st.stop()

    # --------------------------------------------------------
    # User Message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(
            question
        )

    # --------------------------------------------------------
    # Assistant Message
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching the leave policy..."
        ):

            try:

                response = requests.post(
                    f"{BACKEND_URL}/query",
                    json={
                        "question": question
                    },
                    timeout=120,
                )

                # ====================================================
                # Successful Backend Response
                # ====================================================

                if response.status_code == 200:

                    result = response.json()

                    answer = result.get(
                        "answer",
                        "No answer returned.",
                    )

                    sources = result.get(
                        "sources",
                        [],
                    )

                    # ------------------------------------------------
                    # SAFE CONFIDENCE HANDLING
                    # ------------------------------------------------

                    confidence = result.get(
                        "confidence"
                    )

                    if confidence is None:

                        confidence_text = "N/A"

                    else:

                        try:

                            confidence = float(
                                confidence
                            )

                            confidence_text = (
                                f"{confidence:.2f}"
                            )

                        except (
                            TypeError,
                            ValueError
                        ):

                            confidence_text = "N/A"

                    # ------------------------------------------------
                    # Provider
                    # ------------------------------------------------

                    llm_provider = result.get(
                        "llm_provider"
                    )

                    retrieval_method = result.get(
                        "retrieval_method",
                        "legacy_rag",
                    )

                    # ------------------------------------------------
                    # Display Answer
                    # ------------------------------------------------

                    st.markdown(
                        answer
                    )

                    # ------------------------------------------------
                    # Metadata
                    # ------------------------------------------------

                    col1, col2, col3 = st.columns(
                        3
                    )

                    with col1:

                        st.caption(
                            f"Retrieval Confidence: "
                            f"{confidence_text}"
                        )

                    with col2:

                        if llm_provider:

                            st.caption(
                                f"LLM: "
                                f"{llm_provider}"
                            )

                        else:

                            st.caption(
                                "LLM: N/A"
                            )

                    with col3:

                        if retrieval_method == (
                            "agentic_rag"
                        ):

                            st.caption(
                                "🤖 Agentic RAG"
                            )

                        else:

                            st.caption(
                                "📚 Legacy RAG"
                            )

                    # =================================================
                    # Sources
                    # =================================================

                    if sources:

                        with st.expander(
                            "📚 View Sources"
                        ):

                            for source in sources:

                                source_id = (
                                    source.get("id")
                                    or "Unknown"
                                )

                                section = (
                                    source.get("section")
                                    or "Unknown"
                                )

                                source_type = (
                                    source.get("type")
                                    or "Unknown"
                                )

                                st.markdown(
                                    f"""
                                    <div class="source-box">
                                    <b>Document:</b> {source_id}<br>
                                    <b>Section:</b> {section}<br>
                                    <b>Type:</b> {source_type}
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    # =================================================
                    # Save Assistant Message
                    # =================================================

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )

                # ====================================================
                # Backend Error
                # ====================================================

                else:

                    try:

                        error_detail = (
                            response.json().get(
                                "detail",
                                "Unknown backend error.",
                            )
                        )

                    except ValueError:

                        error_detail = response.text

                    st.error(
                        f"Backend error "
                        f"({response.status_code}): "
                        f"{error_detail}"
                    )

            # ========================================================
            # Connection Error
            # ========================================================

            except requests.ConnectionError:

                st.error(
                    "Could not connect to the FastAPI backend. "
                    "Make sure the backend is running."
                )

            # ========================================================
            # Timeout
            # ========================================================

            except requests.Timeout:

                st.error(
                    "The request timed out. "
                    "Please try again."
                )

            # ========================================================
            # Other Request Error
            # ========================================================

            except requests.RequestException as exc:

                st.error(
                    f"Request failed: {exc}"
                )

            # ========================================================
            # Unexpected Error
            # ========================================================

            except Exception as exc:

                st.error(
                    f"Unexpected error: {exc}"
                )