from typing import Any, Iterable, Optional, TypeVar

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    Content,
    ContentText,
)

T = TypeVar("T")


def format_function_call(name: str, kwargs: dict[str, Any], *, maxlen: int = 70) -> str:
    if len(repr(kwargs)) < maxlen:
        return f"{name}({', '.join(f'{k}={v!r}' for k, v in kwargs.items())})"
    result = f"{name}:\n\n"
    for k, v in kwargs.items():
        if len(repr(v)) < maxlen or (isinstance(v, str) and "\n" not in v):
            result += f"{k}: {v!r}\n"
        elif isinstance(v, str) and "\n" in v:
            result += f"{k}:\n'''\n{v}\n'''\n" ""
        else:
            result += f"{k}:\n```\n{v!r}\n```\n" ""
    return result


def fuse_content(*messages: str | list[Content]) -> Iterable[Content]:
    for message in messages:
        if isinstance(message, str):
            yield ContentText(text=message)
        else:
            yield from message


def fuse_tool_calls(*ls: Optional[list[T]]) -> Optional[list[T]]:
    result = None
    for l in ls:
        if l is not None:
            if result is None:
                result = []
            result.extend(l)
    return result


def fuse_similar_messages(
    *messages: ChatMessage | Iterable[ChatMessage],
) -> Iterable[ChatMessage]:
    prev_message: Optional[ChatMessage] = None
    remaining = messages
    while remaining:
        cur_message, *remaining = remaining
        if not isinstance(
            cur_message,
            (ChatMessageSystem, ChatMessageUser, ChatMessageAssistant, ChatMessageTool),
        ) and isinstance(cur_message, Iterable):
            remaining = tuple(cur_message) + tuple(remaining)
            continue
        if prev_message is None:
            prev_message = cur_message
            continue
        if isinstance(prev_message, ChatMessageUser) and isinstance(
            cur_message, ChatMessageUser
        ):
            yield ChatMessageUser(
                content=list(fuse_content(prev_message.content, cur_message.content)),
                source=prev_message.source or cur_message.source,
            )
        elif isinstance(prev_message, ChatMessageSystem) and isinstance(
            cur_message, ChatMessageSystem
        ):
            yield ChatMessageSystem(
                content=list(fuse_content(prev_message.content, cur_message.content)),
                source=prev_message.source or cur_message.source,
            )
        elif isinstance(prev_message, ChatMessageAssistant) and isinstance(
            cur_message, ChatMessageAssistant
        ):
            yield ChatMessageAssistant(
                content=list(fuse_content(prev_message.content, cur_message.content)),
                source=prev_message.source or cur_message.source,
                tool_calls=fuse_tool_calls(
                    prev_message.tool_calls, cur_message.tool_calls
                ),
            )
        else:
            yield prev_message
            prev_message = cur_message


def strip_tool_uses(
    *messages: ChatMessage | Iterable[ChatMessage],
    transform_tool_message_to_user_message: bool = True,
) -> Iterable[ChatMessageSystem | ChatMessageUser | ChatMessageAssistant]:
    """Strip tool uses from messages."""
    for message in messages:
        if isinstance(message, ChatMessageTool):
            if transform_tool_message_to_user_message:
                metadata = {}
                if message.tool_call_id:
                    metadata["tool_call_id"] = message.tool_call_id
                if message.function:
                    metadata["function"] = message.function
                if message.error:
                    metadata["error"] = repr(message.error)
                metadata_str = f" ({metadata})" if metadata else ""
                if isinstance(message.content, str):
                    yield ChatMessageUser(
                        content=f"ChatMessageTool{metadata_str}:\n{message.content}"
                    )
                else:
                    assert isinstance(message.content, list), message.content
                    yield ChatMessageUser(
                        content=[
                            ContentText(text=f"ChatMessageTool{metadata_str}:"),
                            *message.content,
                        ],
                        source="generate",
                    )
        elif isinstance(message, ChatMessageAssistant):
            update: dict = {
                "tool_calls": None,
                "content": (
                    [ContentText(text=message.content)]
                    if isinstance(message.content, str)
                    else list(message.content)
                ),
            }
            if message.tool_calls and transform_tool_message_to_user_message:
                for tool_call in message.tool_calls:
                    update["content"].append(
                        ContentText(
                            text=f"Tool call ({tool_call.id}): {format_function_call(tool_call.function, tool_call.arguments)}"
                        )
                    )
            yield message.model_copy(update=update)
        elif isinstance(
            message, (ChatMessageSystem, ChatMessageUser)
        ) or not isinstance(message, Iterable):
            yield message
        else:
            yield from strip_tool_uses(*message)
