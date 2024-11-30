# %%
from typing import Optional

import inspect_ai.scorer
import inspect_ai.util
from inspect_ai.tool import ToolResult, tool
from inspect_ai.util import Store


@tool
def finished_tool(name: Optional[str] = None):
    key = f"finished {name}" if name else "finished"

    async def run() -> ToolResult:
        """Tool indicating that the (sub)task is finished."""
        inspect_ai.util.store().set(key, True)
        return True

    if name:
        run.__doc__ = f"Tool indicating that the {name} (sub)task is finished."

    return run


def is_finished(name: Optional[str] = None, *, store: Optional[Store] = None) -> bool:
    key = f"finished {name}" if name else "finished"
    if store is None:
        store = inspect_ai.util.store()
    return store.get(key, False)


def reset_finished(
    name: Optional[str] = None, *, store: Optional[Store] = None
) -> None:
    key = f"finished {name}" if name else "finished"
    if store is None:
        store = inspect_ai.util.store()
    store.set(key, False)
