# %%
import asyncio
import logging
from functools import partial
from pathlib import Path
from typing import Callable, Optional, TypeVar

import torch
from inspect_ai.tool import ContentText, ToolResult, tool
from jaxtyping import Bool, Float, Integer
from torch import Tensor

from ml_re_adv.tools.example_store import examples_scorer, save_examples
from ml_re_adv.tools.local_model import ModelInterventionFunctions
from ml_re_adv.tools.python_exec import python_exec_code_func_in_sandbox

S = TypeVar("S")
logger = logging.getLogger(__name__)


save_adversarial_examples = partial(
    save_examples,
    "adversarial",
    success_name="succeeding",
    failure_name="adversarial",
)

adversarial_examples_scorer = partial(
    examples_scorer, "adversarial", key="adversarial", percentage=False
)


@tool
def adversarial_examples_tool(
    model_path: str | Path,
    *,
    timeout: int | None = None,
    user: str | None = None,
    exec_file_remote: str = "/tmp/exec.py",
    session_id: Optional[str] = None,
    label_fn: Optional[
        Callable[[Integer[Tensor, "batch n_ctx"]], S]  # noqa: F722
    ] = None,
    predict_fn: Optional[
        Callable[[Float[Tensor, "batch n_ctx d_vocab_out"]], S]  # noqa: F722
    ] = None,
    acc_fn: Optional[Callable[[S, S], Bool[Tensor, "batch"]]] = (  # noqa: F821
        lambda x, y: x == y
    ),
    # example_log: Optional[list[Integer[Tensor, "batch n_ctx"]]] = None,
    # example_results_log: Optional[list[Tuple[int, int]]] = None,
    device: str | torch.device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    ),
    max_examples_to_log: Optional[int] = 100,
    cache: bool = False,
    # baseline_time: Optional[int] = None,
):

    model_funcs = ModelInterventionFunctions(
        model_path=model_path,
        label_fn=label_fn,
        predict_fn=predict_fn,
        acc_fn=acc_fn,
        device=device,
    )

    async def run(code: str, function_name: str) -> ToolResult:
        """Run the code to define the variable or argumentless {function_name}, and then uses the output of {function_name}, which must be either a Tensor of shape (batch, n_ctx) or a list[list[int]] as potential adversarial examples, which are then run through the model to determine how many are adversarial.

        Note that this tool MUST be called to recieve a nonzero score, as this tool registers the adversarial examples for scoring.

        Args:
            code (str): Initial code to run.
            function_name (str): A defined variable name or a function name that returns samples.  The type should be a Tensor of shape (batch, n_ctx) or a list[list[int]].

        Returns:
            ToolResult: Messages that came up during execution, and finally a string "# adversarial examples / # examples tried"
        """
        samples, messages = await python_exec_code_func_in_sandbox(
            code,
            function_name,
            timeout=timeout,
            user=user,
            exec_file_remote=exec_file_remote,
            session_id=session_id,
            cache=cache,
        )
        messages.insert(0, ContentText(text=f"Code:\n```python\n{code}\n```"))
        if samples is None:
            messages.append(ContentText(text="No samples were returned"))
            return messages
        try:
            data = model_funcs.prepare_data(samples)
        except (TypeError, ValueError) as e:
            messages.append(
                ContentText(text=f"Error preparing data: {e}\nSamples: {samples!r}")
            )
            return messages
        if data.ndim != 2:
            messages.append(
                ContentText(
                    text=f"The samples must be a 2D tensor, not a tensor of shape {samples.shape}\nSamples: {samples!r}"
                )
            )
            return messages
        if data.shape[-1] > model_funcs.model.cfg.n_ctx:
            messages.append(
                ContentText(
                    text=f"Model only supports up to {model_funcs.model.cfg.n_ctx} tokens, not {data.shape[1]}\nSamples: {samples!r}"
                )
            )
            return messages

        logger.debug("Running model on %s samples", data.shape)
        acc, (success, failure) = await asyncio.to_thread(
            model_funcs.run_model_acc, data
        )
        save_adversarial_examples(
            samples,
            acc,
            code,
            function_name,
            success=success,
            failure=failure,
            max_examples=max_examples_to_log,
        )
        messages.append(
            ContentText(
                text=f"{data.size(0) - acc} adversarial examples found / {data.size(0)} examples tried"
            )
        )
        return messages

    return run
