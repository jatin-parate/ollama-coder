"""Agent executor for LangGraph workflow."""

import asyncio
import logging
import threading
import uuid
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple, Union

from langchain_core.caches import BaseCache
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph

from .context_builder import ContextBuilder
from .system_prompt import SystemPromptBuilder
from ..tools.base import BaseTool
from ..tools.registry import ToolRegistry
from ..memory.store import ProjectMemoryStore
from ..memory.extractor import ProjectMemoryExtractor
from ..exceptions import ToolExecutionError, MemoryError

logger = logging.getLogger(__name__)


class AgentExecutor:
    """Executes the LangGraph workflow for chat and tool operations."""

    def __init__(
        self,
        model: ChatOllama,
        tool_registry: ToolRegistry,
        context_builder: ContextBuilder,
        memory: MemorySaver = None,
        project_memory: ProjectMemoryStore = None,
        project_memory_extractor: ProjectMemoryExtractor = None,
        history_token_budget: int = 1024,
    ):
        """Initialize the agent executor.

        Args:
            model: The base language model.
            tool_registry: The tool registry.
            context_builder: The context builder.
            memory: The memory checkpointer for conversation state.
            project_memory: The project memory store.
            project_memory_extractor: The project memory extractor.
            history_token_budget: The token budget for message history.
        """
        self.model = model
        self.tool_registry = tool_registry
        self.context_builder = context_builder
        self.memory = memory or MemorySaver()
        self.project_memory = project_memory
        self.project_memory_extractor = project_memory_extractor
        self.history_token_budget = history_token_budget

        # Bind tools to model
        self.model_with_tools = self.model.bind_tools(
            self.tool_registry.get_langchain_tools()
        )

        # Build workflow
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.memory)

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow.

        Returns:
            The compiled workflow.
        """
        workflow = StateGraph(MessagesState)

        # Add nodes
        workflow.add_node("chat", self._chat_node)
        workflow.add_node("tools", self._tool_node)

        # Set entry point
        workflow.set_entry_point("chat")

        # Add conditional edges with explicit END routing
        workflow.add_conditional_edges(
            "chat",
            self._should_continue,
            {"tools": "tools", "__end__": END}
        )
        workflow.add_edge("tools", "chat")

        return workflow

    def _build_chat_context(self, state: MessagesState) -> tuple[str, List[Any]]:
        """Build chat context from state messages.

        Args:
            state: The current message state.

        Returns:
            Tuple of (system_prompt, processed_messages).
        """
        # Extract active user query
        active_query = self.context_builder.extract_active_query(state["messages"])

        # Build context
        if state["messages"]:
            last_msg = state["messages"][-1]
            if hasattr(last_msg, "content"):
                system_prompt, messages = self.context_builder.build_context(
                    last_msg.content, state["messages"], active_query
                )
            else:
                system_prompt, messages = self.context_builder.build_context(
                    "", state["messages"], active_query
                )
        else:
            system_prompt, messages = self.context_builder.build_context(
                "", [], active_query
            )

        # Trim messages for context
        messages = self._trim_messages(messages)

        return system_prompt, messages

    def _chat_node(self, state: MessagesState) -> Dict[str, List[Any]]:
        """Process user message and get model response.

        Args:
            state: The current message state.

        Returns:
            The updated state with the model response.
        """
        logger.info("=== CHAT NODE START ===")
        logger.info(f"State messages count: {len(state['messages'])}")

        # Build context
        system_prompt, messages = self._build_chat_context(state)

        logger.info(f"Messages prepared for model: {len(messages)}")

        # Invoke model
        response = self.model_with_tools.invoke(messages)
        self._attach_token_metadata(messages, response)

        logger.info(f"Model response type: {type(response)}")
        if hasattr(response, "tool_calls"):
            logger.info(f"Tool calls in response: {response.tool_calls}")

        logger.info("=== CHAT NODE END ===")
        return {"messages": [response]}

    def _stream_chat_node(self, state: MessagesState) -> Iterator[Dict[str, List[Any]]]:
        """Process user message and stream model response.

        Args:
            state: The current message state.

        Yields:
            The updated state with streamed model responses.
        """
        logger.info("=== STREAM CHAT NODE START ===")
        logger.info(f"State messages count: {len(state['messages'])}")

        # Build context
        system_prompt, messages = self._build_chat_context(state)

        logger.info(f"Messages prepared for model: {len(messages)}")

        # Stream model response
        full_content = ""
        tool_calls = []

        try:
            for chunk in self.model_with_tools.stream(messages):
                logger.info(f"Stream chunk: {chunk}")

                # Accumulate content
                if hasattr(chunk, "content") and chunk.content:
                    full_content += chunk.content

                # Accumulate tool calls
                if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                    for tc in chunk.tool_calls:
                        # Check if tool call already exists
                        existing = next(
                            (t for t in tool_calls if t.get("name") == tc.get("name")),
                            None,
                        )
                        if existing:
                            # Merge arguments
                            if "args" in tc and "args" in existing:
                                existing["args"].update(tc["args"])
                        else:
                            tool_calls.append(tc)

                # Yield chunk for streaming
                if hasattr(chunk, "content") and chunk.content:
                    yield {"messages": [chunk]}

            # Create final message with accumulated content and tool calls
            if full_content or tool_calls:
                final_msg = AIMessage(content=full_content, tool_calls=tool_calls)
                self._attach_token_metadata(messages, final_msg)
                logger.info(f"Final streamed message: {final_msg}")
                yield {"messages": [final_msg]}

        except Exception as e:
            logger.error(f"Error streaming model response: {e}")
            # Fall back to invoke
            response = self.model_with_tools.invoke(messages)
            self._attach_token_metadata(messages, response)
            yield {"messages": [response]}

        logger.info("=== STREAM CHAT NODE END ===")

    def _tool_node(self, state: MessagesState) -> Dict[str, List[Any]]:
        """Execute tools requested by the model.

        Args:
            state: The current message state.

        Returns:
            The updated state with tool results.
        """
        logger.info("=== TOOL NODE START ===")

        messages = state["messages"]
        last_message = messages[-1]

        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            logger.info("No tool calls to process")
            logger.info("=== TOOL NODE END ===")
            return {"messages": []}

        tool_results = []
        for tool_call in last_message.tool_calls:
            try:
                result = self._execute_tool_call(tool_call)
                if result is not None:
                    tool_msg = ToolMessage(
                        content=result,
                        tool_call_id=self._extract_tool_call_field(tool_call, "id") or str(uuid.uuid4()),
                    )
                    tool_results.append(tool_msg)
            except ToolExecutionError as e:
                logger.error(f"Tool execution error: {e}")
                tool_msg = ToolMessage(
                    content=f"Error: {str(e)}",
                    tool_call_id=self._extract_tool_call_field(tool_call, "id") or str(uuid.uuid4()),
                )
                tool_results.append(tool_msg)

        logger.info(f"Tool results count: {len(tool_results)}")
        logger.info("=== TOOL NODE END ===")
        return {"messages": tool_results}

    def _execute_tool_call(self, tool_call: Any) -> Optional[str]:
        """Execute a single tool call.

        Args:
            tool_call: The tool call to execute.

        Returns:
            The tool result or None if execution failed.

        Raises:
            ToolExecutionError: If tool execution fails.
        """
        tool_name = self._extract_tool_call_field(tool_call, "name")
        args = self._extract_tool_call_field(tool_call, "args", {})

        logger.info(f"Executing tool: {tool_name} with args: {args}")

        try:
            # Get tool class from registry
            tool_class = self.tool_registry.get_tool(tool_name)

            # Create tool instance
            tool = tool_class(**args)

            # Validate tool
            if not tool.validate():
                logger.warning(f"Tool validation failed: {tool_name}")
                raise ToolExecutionError(f"Tool validation failed for {tool_name}")

            # Execute tool
            result = tool.execute()
            logger.info(f"Tool result: {result[:200]}")
            return result

        except KeyError as e:
            logger.error(f"Tool not found: {tool_name}")
            raise ToolExecutionError(f"Unknown tool: {tool_name}") from e
        except ToolExecutionError:
            raise
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            raise ToolExecutionError(f"Error executing {tool_name}: {str(e)}") from e

    def _should_continue(self, state: MessagesState) -> str:
        """Determine if we should continue to tools or end.

        Args:
            state: The current message state.

        Returns:
            The next node to execute or END.
        """
        messages = state["messages"]
        last_message = messages[-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return END

    def _trim_messages(self, messages: List[Any]) -> List[Any]:
        """Trim message history to preserve context budget.

        Args:
            messages: The message list to trim.

        Returns:
            The trimmed message list.
        """
        if len(messages) <= 1:
            return messages

        try:
            from langchain_core.messages import trim_messages

            trimmed = trim_messages(
                messages,
                max_tokens=self.history_token_budget,
                token_counter=self.model,
                strategy="last",
                start_on="human",
                include_system=False,
            )
            logger.info(
                "Trimmed message history from %s to %s messages",
                len(messages),
                len(trimmed),
            )
            return trimmed
        except Exception as e:
            logger.warning(f"Falling back to untrimmed message history: {e}")
            return messages

    def _extract_tool_call_field(
        self, tool_call: Any, key: str, default: Any = None
    ) -> Any:
        """Extract a field from a tool call.

        Args:
            tool_call: The tool call object or dict.
            key: The field name to extract.
            default: Default value if field not found.

        Returns:
            The field value or default.
        """
        if isinstance(tool_call, dict):
            return tool_call.get(key, default)
        return getattr(tool_call, key, default)

    def _attach_token_metadata(self, messages: List[Any], response: Any) -> None:
        """Attach token metadata to a model response.

        Args:
            messages: The message history.
            response: The model response.
        """
        # Initialize response_metadata if not present
        response_metadata = getattr(response, "response_metadata", None)
        if not isinstance(response_metadata, dict):
            response_metadata = {}
            setattr(response, "response_metadata", response_metadata)

        try:
            full_prompt_tokens = self.model.get_num_tokens_from_messages(messages)
            response_metadata["full_prompt_tokens"] = full_prompt_tokens

            prompt_eval_count = response_metadata.get("prompt_eval_count")
            if isinstance(prompt_eval_count, int):
                response_metadata["cached_prompt_tokens"] = max(
                    full_prompt_tokens - prompt_eval_count, 0
                )
        except Exception as e:
            logger.debug(f"Could not count prompt tokens: {e}")

    def queue_memory_update(
        self, user_input: str, new_messages: List[Any]
    ) -> None:
        """Queue a project memory update in a background thread.

        Args:
            user_input: The original user input.
            new_messages: The new messages from this turn.
        """
        final_ai_messages = [
            msg
            for msg in new_messages
            if getattr(msg, "type", None) == "ai"
            and not getattr(msg, "tool_calls", None)
        ]
        if not final_ai_messages:
            return

        final_answer = getattr(final_ai_messages[-1], "content", "")
        if not str(final_answer).strip():
            return

        # Build transcript
        transcript_parts = [f"User request:\n{user_input}"]
        for msg in new_messages:
            msg_type = getattr(msg, "type", type(msg).__name__)
            msg_content = getattr(msg, "content", "")
            if msg_content:
                transcript_parts.append(f"{msg_type.upper()}:\n{msg_content}")

        transcript = "\n\n".join(transcript_parts)

        def _persist_memory() -> None:
            try:
                if not self.project_memory or not self.project_memory_extractor:
                    return

                facts = self.project_memory_extractor.extract_facts(transcript)
                stored_count = self.project_memory.upsert_facts(facts, transcript)
                logger.info("Stored %s project memory facts", stored_count)
            except MemoryError as e:
                logger.warning(f"Memory error: {e}")
            except Exception as e:
                logger.warning(f"Failed to persist project memory: {e}")

        threading.Thread(target=_persist_memory, daemon=True).start()

    def invoke(self, user_message: str, thread_id: str = "1") -> Dict[str, List[Any]]:
        """Invoke the agent with a user message.

        Args:
            user_message: The user message to process.
            thread_id: The thread ID for conversation state.

        Returns:
            The response messages.
        """
        config = {"configurable": {"thread_id": thread_id}}

        return self.app.invoke(
            {"messages": [("user", user_message)]},
            config=config,
        )

    def stream(
        self, user_message: str, thread_id: str = "1"
    ) -> Iterator[Dict[str, Any]]:
        """Stream the agent response in real-time.

        Args:
            user_message: The user message to process.
            thread_id: The thread ID for conversation state.

        Yields:
            State updates from the agent execution.
        """
        config = {"configurable": {"thread_id": thread_id}}

        try:
            for chunk in self.app.stream(
                {"messages": [("user", user_message)]},
                config=config,
                stream_mode="values",
            ):
                yield chunk
        except Exception as e:
            logger.warning(f"Streaming failed: {e}, falling back to invoke")
            # Fall back to non-streaming
            response = self.invoke(user_message, thread_id)
            yield {"messages": response["messages"]}

    def get_state(self, thread_id: str = "1") -> Any:
        """Get the current conversation state.

        Args:
            thread_id: The thread ID.

        Returns:
            The current state snapshot.
        """
        config = {"configurable": {"thread_id": thread_id}}
        return self.app.get_state(config)
