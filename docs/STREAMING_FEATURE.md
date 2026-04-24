# Real-Time Streaming Feature

## Overview

The CLI now supports real-time streaming of the agent's output, similar to Gemini CLI. Users can see the thinking process and tool calls as they happen, rather than waiting for the complete response.

## How It Works

### Streaming Display

When you send a message, the CLI shows:

1. **Thinking Panel**: A status panel with a spinner that updates in real-time as the model generates its response
2. **Accumulated Content**: The thinking content accumulates and displays as it streams
3. **Tool Calls**: Once the model decides to use tools, they're displayed
4. **Tool Results**: Tool execution results appear after execution

### Example Flow

```
You: What files are in the project?

Thinking: I'll explore the project structure to see what files are present.

Thinking: I'll explore the project structure to see what files are present.
I'll use the BashTool to list the files in the current directory.

Thinking: I'll explore the project structure to see what files are present.
I'll use the BashTool to list the files in the current directory.

Tool Call:
{
  "name": "BashTool",
  "args": {
    "command": "ls -la"
  }
}

Tool Result:
total 832
drwxr-xr-x  17 jparate  staff     544 Apr 24 11:34 .
drwxr-xr-x   8 jparate  staff     256 Apr 23 17:42 ..
...
```

## Implementation Details

### 1. Renderer (`ui/renderer.py`)

The renderer uses a buffer to accumulate streaming content:

```python
def render_streaming_content(self, content: str) -> None:
    """Render streaming content in real-time, similar to Gemini CLI."""
    # Accumulate content
    self._thinking_buffer += content

    # If we don't have a thinking panel yet, create one
    if self._thinking_panel is None:
        self._thinking_panel = self.console.status(
            f"[bold yellow]Thinking...[/bold yellow]",
            spinner="dots",
        )
        self._thinking_panel.start()

    # Update the panel with accumulated content
    if self._thinking_panel:
        self._thinking_panel.update(
            f"[bold yellow]Thinking...[/bold yellow]\n{self._thinking_buffer}",
            spinner="dots",
        )
```

### 2. Finalization

When streaming is complete, the thinking panel is finalized and displayed as a proper panel:

```python
def finalize_thinking(self) -> None:
    """Finalize the thinking panel and display the full content."""
    if self._thinking_panel is not None:
        self._thinking_panel.stop()
        self._thinking_panel = None

        if self._thinking_buffer:
            self.console.print(
                Panel(
                    Markdown(self._thinking_buffer),
                    title="[bold yellow]Thinking[/bold yellow]",
                    title_align="left",
                    border_style="yellow",
                    padding=(1, 2),
                )
            )
        self._thinking_buffer = ""
```

### 3. CLI Integration (`cli.py`)

The CLI streams responses and processes them in real-time:

```python
def _stream_response(
    self, user_input: str, thread_id: str, previous_count: int
) -> None:
    """Stream the agent response in real-time."""
    current_ai_message = None
    current_tool_calls = []

    for chunk in self.agent.stream(user_input, thread_id=thread_id):
        messages = chunk.get("messages", [])

        for msg in messages:
            if msg.type == "ai":
                if not getattr(msg, "tool_calls", None):
                    # Show thinking content as it streams
                    if hasattr(msg, "content") and msg.content:
                        self.console.render_streaming_content(str(msg.content))
                else:
                    current_tool_calls.extend(msg.tool_calls)
            elif msg.type == "tool":
                self.console.render_tool_message(msg)

    # Render final AI message with tool calls
    if current_ai_message:
        self.console.render_assistant_message(current_ai_message)
```

## Benefits

1. **Real-time Feedback**: Users see the agent's thinking as it happens
2. **Gemini-like UX**: Similar to Gemini CLI's streaming behavior
3. **Transparency**: Users can see the agent's reasoning process unfold
4. **Engagement**: More interactive and engaging experience
5. **Fallback**: If streaming fails, falls back to non-streaming mode

## Technical Details

### Streaming Mode

The agent uses LangGraph's `stream()` method with `stream_mode="values"`:

```python
for chunk in self.app.stream(
    {"messages": [("user", user_message)]},
    config=config,
    stream_mode="values",
):
    yield chunk
```

### Content Accumulation

Streaming content is accumulated in a buffer and displayed in a status panel:

1. Start with a status panel showing "Thinking..."
2. Update the panel as new content arrives
3. Finalize and display as a proper panel when done

### Tool Call Handling

Tool calls are collected during streaming and displayed after the thinking content is finalized.

## Future Enhancements

1. **Markdown Streaming**: Render markdown as it streams (line by line)
2. **Code Block Streaming**: Show code blocks as they're generated
3. **Progress Indicators**: Show token count and progress
4. **Cancellable Streaming**: Allow users to cancel long-running streams
5. **Syntax Highlighting**: Add syntax highlighting to streaming code blocks

