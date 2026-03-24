"""
app.py

Streamlit UI for the Bluesky Image Post Agent.

Run:
    streamlit run app.py

What this UI does:
  - Step 1  : Enter topic + audience, click Run
  - Step 2  : See fetched articles, tick the ones you want, click Approve
  - Step 3  : Watch the progress bar while hook, content, and image generate
  - Step 4  : See the generated image + full post preview side by side
              Approve / Edit (with instructions) / Reject
  - Step 5  : See the published post URL with a clickable link

All agent state lives in st.session_state so the UI survives Streamlit reruns.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from langgraph.types import Command

load_dotenv()

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Bluesky Image Post Agent",
    page_icon="🦋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import agent (after page config) ─────────────────────────────────────────
from main import BlueskyAgent
from config.settings import WorkflowStatus, workflow_cfg


# ── Session state helpers ─────────────────────────────────────────────────────

def _init_state() -> None:
    """Initialise all session state keys on first load."""
    defaults = {
        "agent":          None,
        "run_id":         None,
        "result":         None,
        "phase":          "setup",      # setup | news_review | generating | approval | done | error
        "log":            [],
        "articles":       [],
        "final_post":     None,
        "image_path":     None,
        "publish_receipt": None,
        "error_msg":      None,
        "retry_count":    0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _log(msg: str) -> None:
    st.session_state.log.append(msg)


def _reset() -> None:
    """Full reset back to setup screen."""
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    _init_state()


# ── Agent helpers ─────────────────────────────────────────────────────────────

def _get_agent() -> BlueskyAgent:
    if st.session_state.agent is None:
        st.session_state.agent = BlueskyAgent()
    return st.session_state.agent


def _extract_interrupt(result: dict) -> Optional[dict]:
    payload = result.get("__interrupt__")
    if payload:
        return payload[0].value
    return None


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:
        st.title("🦋 Bluesky Agent")
        st.caption("AI-powered image post generator")

        st.divider()

        # Env status
        st.subheader("Environment")
        checks = {
            "GROQ_API_KEY": "Groq LLM",
            "TAVILY_API_KEY":    "Tavily Search",
            "GRADIO_URL":        "Colab GPU",
            "BLUESKY_HANDLE":    "Bluesky Handle",
            "BLUESKY_APP_PASSWORD": "Bluesky Password",
        }
        all_ok = True
        for key, label in checks.items():
            val = os.environ.get(key, "")
            if val:
                st.success(f"{label}", icon="✅")
            else:
                st.error(f"{label} missing", icon="❌")
                all_ok = False

        if not all_ok:
            st.warning("Fill in your .env file before running.", icon="⚠️")

        st.divider()

        # Current phase
        st.subheader("Status")
        phase_labels = {
            "setup":       "⬜ Waiting to start",
            "news_review": "📰 Reviewing news",
            "generating":  "⚙️ Generating content",
            "approval":    "👀 Awaiting approval",
            "done":        "✅ Published",
            "error":       "❌ Error occurred",
        }
        st.info(phase_labels.get(st.session_state.phase, "Unknown"))

        st.divider()

        # Activity log
        st.subheader("Activity Log")
        if st.session_state.log:
            for entry in reversed(st.session_state.log[-10:]):
                st.caption(entry)
        else:
            st.caption("Nothing yet.")

        st.divider()

        if st.button("🔄 Start Over", use_container_width=True):
            _reset()
            st.rerun()


# ── Phase: Setup ──────────────────────────────────────────────────────────────

def _render_setup() -> None:
    st.title("🦋 Bluesky Image Post Agent")
    st.caption("Search news → Generate post + AI image → Review → Post to Bluesky")

    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        topic = st.text_input(
            "Topic",
            placeholder="e.g. AI Agents 2026, Climate Tech, Rust programming",
            help="The subject to search news for. Be specific for better results.",
        )

    with col2:
        audience = st.text_input(
            "Target Audience (optional)",
            placeholder="e.g. AI developers, startup founders",
            help="Who you are writing for. Shapes the tone and hook.",
        )

    st.divider()

    col_run, col_spacer = st.columns([1, 3])
    with col_run:
        run_clicked = st.button(
            "🔍 Search & Generate",
            type="primary",
            use_container_width=True,
            disabled=not topic.strip(),
        )

    if run_clicked and topic.strip():
        run_id = f"{workflow_cfg.graph_thread_id_prefix}_{uuid.uuid4().hex[:8]}"
        st.session_state.run_id = run_id

        _log(f"Starting run: {run_id}")
        _log(f"Topic: {topic}")

        with st.spinner("Searching for latest news..."):
            agent  = _get_agent()
            result = agent.start(
                run_id=run_id,
                topic=topic.strip(),
                target_audience=audience.strip() or None,
            )

        st.session_state.result = result
        interrupt = _extract_interrupt(result)

        if interrupt and "articles" in interrupt:
            st.session_state.articles    = interrupt["articles"]
            st.session_state.retry_count = interrupt.get("retry_count", 0)
            st.session_state.phase       = "news_review"
            _log(f"Fetched {len(interrupt['articles'])} articles")
            st.rerun()
        else:
            status = agent.get_workflow_status(run_id)
            st.session_state.phase    = "error"
            st.session_state.error_msg = f"Search failed. Status: {status}"
            _log(f"Error: {st.session_state.error_msg}")
            st.rerun()


# ── Phase: News Review ────────────────────────────────────────────────────────

def _render_news_review() -> None:
    st.title("📰 Review News Articles")
    st.caption("Select the articles you want to base the post on, then click Approve.")

    agent    = _get_agent()
    run_id   = st.session_state.run_id
    articles = st.session_state.articles
    retry    = st.session_state.retry_count

    if retry > 0:
        st.info(f"Search retry {retry} of {workflow_cfg.max_search_retries}", icon="🔁")

    if not articles:
        st.error("No articles found. Try a different topic.")
        if st.button("← Back to Setup"):
            _reset()
            st.rerun()
        return

    st.divider()

    # Show articles as cards with checkboxes
    selected = []
    for a in articles:
        col_check, col_content = st.columns([0.5, 9.5])
        with col_check:
            checked = st.checkbox("", key=f"article_{a['index']}", value=True)
        with col_content:
            st.markdown(f"**[{a['title']}]({a['url']})**")
            st.caption(f"Source: {a['source']}")
        if checked:
            selected.append(a["index"])

    st.divider()

    col_approve, col_reject = st.columns([2, 1])

    with col_approve:
        approve_disabled = len(selected) == 0
        if st.button(
            f"✅ Approve {len(selected)} article(s) and Generate",
            type="primary",
            use_container_width=True,
            disabled=approve_disabled,
        ):
            _log(f"Approved articles: {selected}")
            st.session_state.phase = "generating"

            with st.spinner("Generating hook, post content, and image... this takes ~60 seconds"):
                result = agent.resume(
                    run_id,
                    Command(resume={"action": "approve", "approved_indices": selected}),
                )

            st.session_state.result = result
            interrupt = _extract_interrupt(result)

            if interrupt and "post" in interrupt:
                # Reached human_approval - get image and post from state
                state = agent.get_state(run_id)
                st.session_state.final_post = state.get("final_post")
                st.session_state.image_path = state.get("image_path")
                st.session_state.phase      = "approval"
                _log("Content and image generated. Awaiting approval.")
                st.rerun()
            else:
                status = agent.get_workflow_status(run_id)
                st.session_state.phase     = "error"
                st.session_state.error_msg = f"Generation failed. Status: {status}"
                _log(f"Error: {st.session_state.error_msg}")
                st.rerun()

    with col_reject:
        if st.button("🔄 Reject All & Re-search", use_container_width=True):
            _log("Rejected all articles. Re-searching...")
            with st.spinner("Re-searching..."):
                result = agent.resume(run_id, Command(resume={"action": "reject"}))

            st.session_state.result = result
            interrupt = _extract_interrupt(result)

            if interrupt and "articles" in interrupt:
                st.session_state.articles    = interrupt["articles"]
                st.session_state.retry_count = interrupt.get("retry_count", 0)
                st.session_state.phase       = "news_review"
                _log(f"New search: {len(interrupt['articles'])} articles")
                st.rerun()
            else:
                status = agent.get_workflow_status(run_id)
                st.session_state.phase     = "error"
                st.session_state.error_msg = f"Max retries reached or search failed. Status: {status}"
                _log(f"Error: {st.session_state.error_msg}")
                st.rerun()


# ── Phase: Approval ───────────────────────────────────────────────────────────

def _render_approval() -> None:
    st.title("👀 Review Post & Image")
    st.caption("Check the generated image and post text. Approve to publish, Edit to refine, or Reject to discard.")

    agent      = _get_agent()
    run_id     = st.session_state.run_id
    final_post = st.session_state.final_post or {}
    image_path = st.session_state.image_path

    st.divider()

    # Two column layout: image on left, post text on right
    col_img, col_post = st.columns([1, 1])

    with col_img:
        st.subheader("Generated Image")
        if image_path and Path(image_path).exists():
            st.image(image_path, use_container_width=True)
            st.caption(f"Saved at: `{image_path}`")
        else:
            st.warning("Image not found or not generated yet.", icon="⚠️")
            st.caption(f"Expected path: {image_path}")

    with col_post:
        st.subheader("Post Preview")

        caption   = final_post.get("caption", "")
        char_count = final_post.get("char_count", len(caption))
        hashtags  = agent.get_state(run_id).get("hashtags") or []

        # Show full post text in a readable box
        st.text_area(
            "Caption",
            value=caption,
            height=250,
            disabled=True,
            key="post_preview",
        )

        # Character count with colour indicator
        colour = "normal" if char_count <= 280 else "inverse"
        st.metric(
            "Character count",
            f"{char_count} / 300",
            delta=f"{300 - char_count} remaining",
            delta_color="normal" if char_count <= 280 else "inverse",
        )

        if hashtags:
            st.caption("Hashtags: " + " ".join(hashtags))

    st.divider()

    # Also show image prompt for transparency
    image_prompt = agent.get_state(run_id).get("image_prompt", "")
    if image_prompt:
        with st.expander("Image prompt used for generation"):
            st.code(image_prompt, language=None)

    st.divider()

    # Action buttons
    st.subheader("Your Decision")

    col_approve, col_edit, col_reject = st.columns([2, 2, 1])

    with col_approve:
        if st.button("🚀 Approve & Post to Bluesky", type="primary", use_container_width=True):
            _log("Approved. Publishing to Bluesky...")
            with st.spinner("Posting to Bluesky..."):
                result = agent.resume(run_id, Command(resume={"action": "approve"}))

            receipt = agent.get_publish_receipt(run_id)
            status  = agent.get_workflow_status(run_id)

            if status == WorkflowStatus.PUBLISHED_SUCCESSFULLY and receipt:
                st.session_state.publish_receipt = receipt
                st.session_state.phase           = "done"
                _log(f"Published: {receipt['post_url']}")
                st.rerun()
            else:
                err = agent.get_state(run_id).get("error_message", "Unknown error")
                st.session_state.phase     = "error"
                st.session_state.error_msg = err
                _log(f"Publish failed: {err}")
                st.rerun()

    with col_edit:
        with st.expander("✏️ Edit instructions"):
            edit_instructions = st.text_area(
                "What should change?",
                placeholder="e.g. make the hook more direct, use simpler language, focus more on the stats",
                height=100,
                key="edit_instructions_input",
            )
            if st.button("🔄 Regenerate with edits", use_container_width=True, disabled=not edit_instructions.strip()):
                _log(f"Editing: {edit_instructions}")
                with st.spinner("Regenerating post and image..."):
                    result = agent.resume(
                        run_id,
                        Command(resume={"action": "edit", "instructions": edit_instructions.strip()}),
                    )

                st.session_state.result = result
                interrupt = _extract_interrupt(result)

                if interrupt and "post" in interrupt:
                    state = agent.get_state(run_id)
                    st.session_state.final_post = state.get("final_post")
                    st.session_state.image_path = state.get("image_path")
                    _log("Regenerated. Review the updated post and image.")
                    st.rerun()
                else:
                    status = agent.get_workflow_status(run_id)
                    st.session_state.phase     = "error"
                    st.session_state.error_msg = f"Regeneration failed. Status: {status}"
                    _log(f"Error: {st.session_state.error_msg}")
                    st.rerun()

    with col_reject:
        if st.button("🗑️ Reject", use_container_width=True):
            agent.resume(run_id, Command(resume={"action": "reject"}))
            st.session_state.phase = "setup"
            _log("Post rejected and discarded.")
            st.info("Post discarded. You can start a new run.")
            _reset()
            st.rerun()


# ── Phase: Done ───────────────────────────────────────────────────────────────

def _render_done() -> None:
    st.title("✅ Posted Successfully!")

    receipt    = st.session_state.publish_receipt or {}
    post_url   = receipt.get("post_url", "")
    post_uri   = receipt.get("post_uri", "")
    image_path = st.session_state.image_path
    final_post = st.session_state.final_post or {}

    st.success("Your post is now live on Bluesky.", icon="🦋")

    st.divider()

    col_img, col_info = st.columns([1, 1])

    with col_img:
        if image_path and Path(image_path).exists():
            st.image(image_path, use_container_width=True, caption="Published image")

    with col_info:
        st.subheader("Post details")
        if post_url:
            st.link_button("Open post on Bluesky", post_url, use_container_width=True)
        st.caption(f"Post URI: `{post_uri}`")
        st.caption(f"Image saved at: `{image_path}`")

        st.divider()
        st.subheader("Post text")
        st.text_area(
            "",
            value=final_post.get("caption", ""),
            height=200,
            disabled=True,
        )

    st.divider()

    if st.button("🔄 Create Another Post", type="primary"):
        _reset()
        st.rerun()


# ── Phase: Error ──────────────────────────────────────────────────────────────

def _render_error() -> None:
    st.title("❌ Something went wrong")

    st.error(st.session_state.error_msg or "An unknown error occurred.", icon="❌")

    st.divider()

    st.markdown("**Common fixes:**")
    st.markdown("- Check that your Colab notebook is still running and update `GRADIO_URL` in `.env`")
    st.markdown("- Check your API keys are correct in `.env`")
    st.markdown("- Check the Activity Log in the sidebar for more detail")

    if st.button("🔄 Start Over", type="primary"):
        _reset()
        st.rerun()


# ── Main router ───────────────────────────────────────────────────────────────

def main() -> None:
    _init_state()
    _render_sidebar()

    phase = st.session_state.phase

    if phase == "setup":
        _render_setup()
    elif phase == "news_review":
        _render_news_review()
    elif phase == "generating":
        st.title("⚙️ Generating...")
        st.info("Content is being generated. This should update automatically.")
    elif phase == "approval":
        _render_approval()
    elif phase == "done":
        _render_done()
    elif phase == "error":
        _render_error()
    else:
        _render_setup()


if __name__ == "__main__":
    main()