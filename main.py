"""
main.py

The wiring file for the Bluesky Image Post Agent.

Graph flow (8 nodes):
  search_news
    -> human_review_news  (HITL #1 - pick articles)
    -> generate_hook      (Bluesky hook with scoring)
    -> generate_content   (post text + SDXL image prompt)
    -> generate_image     (SDXL via Colab Gradio)
    -> human_approval     (HITL #2 - review post + image)
    -> publish_to_bluesky
    -> error_handler (on any failure)

Before running:
  - Start the Google Colab notebook to get your GRADIO_URL
  - Fill in .env with all required keys
  - Run: python main.py --topic "AI Agents 2026"

Python API:
    from main import BlueskyAgent
    from langgraph.types import Command

    agent  = BlueskyAgent()
    run_id = "run_001"

    result = agent.start(run_id, topic="AI Agents 2026")
    result = agent.resume(run_id, Command(resume={"action": "approve", "approved_indices": [0, 2]}))
    result = agent.resume(run_id, Command(resume={"action": "approve"}))
    receipt = agent.get_publish_receipt(run_id)
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from typing import List, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from config.settings import (
    ApprovalStatus,
    ReviewStatus,
    WorkflowConfig,
    WorkflowStatus,
    workflow_cfg,
)
from core.state import BlueskyAgentState
from edges.routing import route_after_generation, route_after_publish, route_after_search
from nodes.error_handler import ErrorHandlerNode
from nodes.generate_content import GenerateContentNode
from nodes.generate_hook import GenerateHookNode
from nodes.generate_image import GenerateImageNode
from nodes.human_approval import HumanApprovalNode
from nodes.human_review_news import HumanReviewNewsNode
from nodes.publish_to_bluesky import PublishToBlueskyNode
from nodes.search_news import SearchNewsNode
from tools.bluesky_tools import BlueskyPublisher
from tools.image_generator import ImageGeneratorTool
from tools.search_tools import SearchTool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class BlueskyAgent:
    """
    Main agent class. Wraps the LangGraph graph and exposes a clean API.

    Both HITL nodes pause themselves via interrupt() internally.
    Resume with agent.resume(run_id, Command(resume={...})).
    """

    def __init__(
        self,
        search_tool:  Optional[SearchTool]         = None,
        image_tool:   Optional[ImageGeneratorTool] = None,
        publisher:    Optional[BlueskyPublisher]   = None,
        wf_cfg:       WorkflowConfig               = workflow_cfg,
    ) -> None:
        self._wf_cfg       = wf_cfg
        self._checkpointer = MemorySaver()
        self._graph        = self._build_graph(search_tool, image_tool, publisher)
        logger.info("BlueskyAgent | graph compiled and ready")

    def _build_graph(
        self,
        search_tool:  Optional[SearchTool],
        image_tool:   Optional[ImageGeneratorTool],
        publisher:    Optional[BlueskyPublisher],
    ) -> StateGraph:

        node_search       = SearchNewsNode(search_tool=search_tool)
        node_hitl_news    = HumanReviewNewsNode()
        node_hook         = GenerateHookNode()
        node_content      = GenerateContentNode()
        node_image        = GenerateImageNode(image_tool=image_tool)
        node_hitl_approve = HumanApprovalNode()
        node_publish      = PublishToBlueskyNode(publisher=publisher)
        node_error        = ErrorHandlerNode()

        builder = StateGraph(BlueskyAgentState)

        builder.add_node("search_news",         node_search)
        builder.add_node("human_review_news",   node_hitl_news)
        builder.add_node("generate_hook",       node_hook)
        builder.add_node("generate_content",    node_content)
        builder.add_node("generate_image",      node_image)
        builder.add_node("human_approval",      node_hitl_approve)
        builder.add_node("publish_to_bluesky",  node_publish)
        builder.add_node("error_handler",       node_error)

        builder.add_edge(START, "search_news")

        builder.add_conditional_edges(
            "search_news",
            route_after_search,
            {"human_review_news": "human_review_news", "error_handler": "error_handler"},
        )

        # human_review_news routes itself via Command:
        #   approve -> "generate_hook"  (sequential chain starts)
        #   reject  -> "search_news"
        #   cap hit -> "error_handler"

        # Sequential chain: hook -> content -> image
        builder.add_edge("generate_hook",    "generate_content")
        builder.add_edge("generate_content", "generate_image")

        builder.add_conditional_edges(
            "generate_image",
            route_after_generation,
            {"human_approval": "human_approval", "error_handler": "error_handler"},
        )

        # human_approval routes itself via Command:
        #   approve -> "publish_to_bluesky"
        #   edit    -> "generate_content"  (regenerate post + image)
        #   reject  -> "__end__"

        builder.add_conditional_edges(
            "publish_to_bluesky",
            route_after_publish,
            {"error_handler": "error_handler", END: END},
        )

        builder.add_edge("error_handler", END)

        return builder.compile(checkpointer=self._checkpointer)

    # Public API

    def start(
        self,
        run_id:          str,
        topic:           str,
        target_audience: Optional[str] = None,
    ) -> dict:
        """
        Start a new run. Returns when the graph pauses at human_review_news.
        The returned dict has a '__interrupt__' key with the articles payload.
        """
        initial_state: BlueskyAgentState = {
            "topic":              topic,
            "target_audience":    target_audience,
            "raw_news":           None,
            "approved_news":      None,
            "news_review_status": ReviewStatus.PENDING,
            "search_retry_count": 0,
            "hook":           None,
            "post_caption":   None,
            "hashtags":       None,
            "image_prompt":   None,
            "final_post":     None,
            "image_path":     None,
            "approval_status": ApprovalStatus.PENDING,
            "human_edits":     None,
            "publish_receipt": None,
            "error_message":   None,
            "workflow_status": WorkflowStatus.STARTED,
        }

        config = self._config(run_id)
        logger.info("BlueskyAgent.start | run_id=%s | topic=%r", run_id, topic)
        return self._run(initial_state, config)

    def resume(self, run_id: str, command: Command) -> dict:
        """
        Resume a paused graph.

        HITL #1 (news review):
            agent.resume(run_id, Command(resume={"action": "approve", "approved_indices": [0, 1]}))
            agent.resume(run_id, Command(resume={"action": "reject"}))

        HITL #2 (final approval):
            agent.resume(run_id, Command(resume={"action": "approve"}))
            agent.resume(run_id, Command(resume={"action": "edit", "instructions": "..."}))
            agent.resume(run_id, Command(resume={"action": "reject"}))
        """
        config = self._config(run_id)
        logger.info("BlueskyAgent.resume | run_id=%s", run_id)
        return self._run(command, config)

    # State inspection helpers

    def get_state(self, run_id: str) -> dict:
        return self._graph.get_state(self._config(run_id)).values

    def get_raw_news(self, run_id: str) -> list:
        return self.get_state(run_id).get("raw_news") or []

    def get_final_post(self, run_id: str) -> Optional[dict]:
        return self.get_state(run_id).get("final_post")

    def get_image_path(self, run_id: str) -> Optional[str]:
        return self.get_state(run_id).get("image_path")

    def get_publish_receipt(self, run_id: str) -> Optional[dict]:
        return self.get_state(run_id).get("publish_receipt")

    def get_workflow_status(self, run_id: str) -> str:
        return self.get_state(run_id).get("workflow_status", "unknown")

    # Internal helpers

    def _config(self, run_id: str) -> dict:
        return {"configurable": {"thread_id": run_id}}

    def _run(self, input_, config: dict) -> dict:
        last_event = {}
        for event in self._graph.stream(input_, config, stream_mode="values"):
            last_event = event
            status = event.get("workflow_status", "")
            if status:
                logger.info("  -> workflow_status: %s", status)
        return last_event


# Interactive CLI

def _display_news(news_items: list) -> None:
    print("\n" + "-" * 60)
    print("  FETCHED NEWS ARTICLES")
    print("-" * 60)
    for i, item in enumerate(news_items):
        print(f"  [{i}] {item['title']}")
        print(f"       Source : {item['source']}")
        print(f"       URL    : {item['url']}")
        print()


def _display_post(final_post: dict, image_path: Optional[str]) -> None:
    print("\n" + "-" * 60)
    print("  BLUESKY POST DRAFT")
    print("-" * 60)
    print(f"\n{final_post.get('caption', '')}")
    print(f"\n  Character count : {final_post.get('char_count', 0)} / 300")
    print(f"  Image path      : {image_path or 'not generated'}")
    print()


def run_interactive_cli(args: argparse.Namespace) -> None:
    agent  = BlueskyAgent()
    run_id = f"{workflow_cfg.graph_thread_id_prefix}_{uuid.uuid4().hex[:8]}"

    print(f"\n{'=' * 60}")
    print(f"  Bluesky Image Post Agent")
    print(f"  Run ID   : {run_id}")
    print(f"  Topic    : {args.topic}")
    print(f"  Audience : {args.audience or 'not specified'}")
    print(f"{'=' * 60}\n")

    print("  Searching for latest news...")
    result = agent.start(run_id=run_id, topic=args.topic, target_audience=args.audience)

    # Phase 2: news review loop
    while True:
        interrupt_payload = result.get("__interrupt__")
        if not interrupt_payload:
            break

        articles = interrupt_payload[0].value.get("articles", [])
        retry    = interrupt_payload[0].value.get("retry_count", 0)

        if not articles:
            print("  No articles found. Exiting.")
            sys.exit(1)

        _display_news(agent.get_raw_news(run_id))

        if retry > 0:
            print(f"  (Retry {retry}/{workflow_cfg.max_search_retries})")

        raw_input = input(
            "  Enter article numbers to APPROVE (e.g. 0,1,2)\n"
            "  or press ENTER to reject and re-search: "
        ).strip()

        if not raw_input:
            print("\n  Rejecting - re-searching...\n")
            result = agent.resume(run_id, Command(resume={"action": "reject"}))
            if not result.get("__interrupt__"):
                if agent.get_workflow_status(run_id) in (WorkflowStatus.ERROR, WorkflowStatus.ERROR_HANDLED):
                    print("  Max retries reached. Exiting.")
                    sys.exit(1)
            continue

        try:
            indices = [int(x.strip()) for x in raw_input.split(",")]
        except ValueError:
            print("  Invalid input.\n")
            continue

        print(f"\n  Approving: {indices}")
        result = agent.resume(run_id, Command(resume={"action": "approve", "approved_indices": indices}))
        break

    # Phase 3: generation runs automatically
    print("\n  Generating hook...")
    print("  Generating post content...")
    print("  Generating image via Colab GPU...\n")

    # Phase 4: final approval loop
    while True:
        interrupt_payload = result.get("__interrupt__")
        if not interrupt_payload:
            break

        final_post = agent.get_final_post(run_id)
        image_path = agent.get_image_path(run_id)

        if not final_post:
            print("  No post generated. Exiting.")
            sys.exit(1)

        _display_post(final_post, image_path)

        if image_path:
            print(f"  Open the image to review: {image_path}\n")

        decision = input(
            "  [A] Approve and post to Bluesky\n"
            "  [E] Edit (provide instructions, will regenerate)\n"
            "  [R] Reject and discard\n"
            "  Choice: "
        ).strip().upper()

        if decision == "A":
            result = agent.resume(run_id, Command(resume={"action": "approve"}))
            break
        elif decision == "E":
            edits = input("  Edit instructions: ").strip()
            if edits:
                result = agent.resume(run_id, Command(resume={"action": "edit", "instructions": edits}))
                print("\n  Regenerating post and image...\n")
                continue
            else:
                print("  No instructions given.\n")
        elif decision == "R":
            agent.resume(run_id, Command(resume={"action": "reject"}))
            print("\n  Post discarded.\n")
            sys.exit(0)
        else:
            print("  Invalid choice.\n")

    # Phase 5: result
    receipt = agent.get_publish_receipt(run_id)
    status  = agent.get_workflow_status(run_id)

    if status == WorkflowStatus.PUBLISHED_SUCCESSFULLY and receipt:
        print(f"\n{'=' * 60}")
        print("  POSTED TO BLUESKY SUCCESSFULLY!")
        print(f"{'=' * 60}")
        print(f"  Post URL : {receipt['post_url']}")
        print(f"  Post URI : {receipt['post_uri']}")
        print()
    else:
        print(f"\n  Ended with status: {status}")
        state = agent.get_state(run_id)
        if state.get("error_message"):
            print(f"  Error: {state['error_message']}")
        sys.exit(1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bluesky Image Post Agent")
    parser.add_argument("--topic",    required=True, help="Topic to post about.")
    parser.add_argument("--audience", default=None,  help="Target audience description.")
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args   = parser.parse_args()
    run_interactive_cli(args)