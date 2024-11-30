# %%
import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import inspect_ai.scorer
import inspect_ai.util
import numpy as np
import pandas as pd
import torch
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState
from inspect_ai.tool import ContentText, ToolError, ToolResult, tool
from inspect_ai.util import Store
from jaxtyping import Float, Integer
from torch import Tensor

from ml_re_adv.tools.local_model import ModelInterventionFunctions
from ml_re_adv.tools.python_exec import python_exec_code_func_in_sandbox
from ml_re_adv.utils import metrics
from ml_re_adv.utils.extra_inspect import getsource_recursively

S = TypeVar("S")
logger = logging.getLogger(__name__)


def init_compression_store(store: Optional[Store] = None) -> None:
    if store is None:
        store = inspect_ai.util.store()
    if __file__ not in store:
        store.set(__file__, [])


def get_compression_attempts(
    store: Optional[Store] = None,
) -> list[dict[str, Any]]:
    if store is None:
        store = inspect_ai.util.store()
    init_compression_store(store)
    return store.get(__file__)


@torch.no_grad()
def save_compression_attempt(store: Optional[Store] = None, **kwargs) -> None:
    if store is None:
        store = inspect_ai.util.store()
    init_compression_store(store)
    get_compression_attempts(store).append(kwargs)


@scorer(
    metrics=[
        inspect_ai.scorer.mean(),
        inspect_ai.scorer.std(),
        inspect_ai.scorer.stderr(),
        metrics.median(),
        metrics.min(),
        metrics.max(),
    ]
)
def compression_scorer() -> Scorer:
    async def run(
        state: TaskState,
        target: Target,
    ) -> Score:
        """
        A scorer to score the results of the examples tool.
        """
        examples = get_compression_attempts(state.store)
        if not examples:
            return Score(value=0, explanation="No examples submitted by tool")
        examples = [
            example
            | {
                "num_parameters_remaining": sum(
                    v.numel() for v in example["synthetic_model_matrices"].values()
                ),
                "num_parameters_total": sum(
                    v.numel() for v in example["model_matrices"].values()
                ),
                "norm_remaining": sum(
                    v.norm() ** 2 for v in example["synthetic_model_matrices"].values()
                )
                .sqrt()
                .item(),
                "norm_full": sum(
                    v.norm() ** 2 for v in example["model_matrices"].values()
                )
                .sqrt()
                .item(),
                "code_len": len(
                    example["synthetic_full_code"].replace(" ", "").replace("\n", "")
                ),
            }
            for example in examples
        ]

        def key(example):
            ce_model_model_matrices = example["ce_model_model_matrices"]
            ce_model_matrices_synthetic = example["ce_model_matrices_synthetic"]
            num_parameters_remaining = example["num_parameters_remaining"]
            num_parameters_total = example["num_parameters_total"]
            norm_remaining = example["norm_remaining"]
            norm_full = example["norm_full"]
            code_len = example["code_len"]
            return (
                num_parameters_remaining / num_parameters_total,
                norm_remaining / norm_full,
                ce_model_model_matrices,
                ce_model_matrices_synthetic,
                code_len,
            )

        def reduce_key(
            num_parameters_remaining_frac,
            norm_remaining_frac,
            ce_model_model_matrices,
            ce_model_matrices_synthetic,
            code_len,
        ):
            return (
                (1 - num_parameters_remaining_frac)
                * (1 - norm_remaining_frac)
                * ce_model_model_matrices
                * ce_model_matrices_synthetic
                / np.log2(code_len)
            )

        best = min(examples, key=key)
        keys = [
            "ce_model_model_matrices",
            "ce_model_matrices_synthetic",
            "num_parameters_remaining",
            "num_parameters_total",
            "norm_remaining",
            "norm_full",
            "code_len",
        ]
        explanation = (
            pd.DataFrame(
                [
                    {k: example[k] for k in keys} | {"score": reduce_key(*key(example))}
                    for example in examples
                ],
                columns=keys,
            ).to_markdown()
            if examples
            else None
        )
        code_keys = [
            "model_matrices_code",
            "model_matrices_function_name",
            "synthetic_model_matrices_code",
            "synthetic_model_matrices_function_name",
            "compute_model_from_matrices_code",
            "compute_model_from_matrices_function_name",
        ]
        return Score(
            value=reduce_key(*key(best)),
            answer="\n\n".join(f"{k}:\n```python\n{best[k]}\n```" for k in code_keys),
            explanation=explanation,
            metadata={k: v for k, v in best.items() if k not in code_keys},
        )

    return run


def check_tensor(x) -> bool:
    return isinstance(x, (torch.Tensor, float, int, np.ndarray)) or (
        isinstance(x, (list, tuple)) and all(check_tensor(xi) for xi in x)
    )


def tensorify(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if isinstance(x, (int, float)):
        return torch.tensor(x)
    if check_tensor(x):
        try:
            return torch.tensor(x)
        except Exception as e:
            raise ValueError(f"Could not convert {x} to a tensor") from e
    raise TypeError(f"Could not convert {x} ({type(x)}) to a tensor")


def sanitize_model_matrices(model_matrices, messages, descr="model matrices"):
    """avoid passing information via anything other than tensors"""
    if model_matrices is None:
        messages.append(ContentText(text=f"No {descr} were returned"))
        return {}, messages
    if not isinstance(model_matrices, dict):
        messages.append(
            ContentText(
                text=f"{descr} must be a dictionary, not {type(model_matrices)}"
            )
        )
        model_matrices = {}
    if not all(isinstance(k, str) for k in model_matrices.keys()):
        non_str_keys = tuple(k for k in model_matrices.keys() if not isinstance(k, str))
        messages.append(
            ContentText(text=f"{descr} keys must be strings, not {non_str_keys}")
        )
        model_matrices = {k: v for k, v in model_matrices.items() if isinstance(k, str)}
    for k, v in list(model_matrices.items()):
        try:
            model_matrices[k] = tensorify(v)
        except (ValueError, TypeError) as e:
            messages.append(
                ContentText(
                    text=f"{descr} (key {k}) values must be Tensors, not {type(v)} ({e})"
                )
            )
            del model_matrices[k]
    return model_matrices, messages


@tool
def compression_tool(
    model_path: str | Path,
    *,
    timeout: int | None = None,
    user: str | None = None,
    exec_file_remote: str = "/tmp/exec.py",
    session_id: Optional[str] = None,
    safe_sandbox_name: Optional[str] = "without_model",
    half_safe_sandbox_name: Optional[str] = "without_model2",
    reduce_fn: Optional[
        Callable[[Integer[Tensor, "batch n_ctx d_vocab_out"]], S]  # noqa: F722
    ] = None,
    compare_fn: Optional[Callable[[S, S], Float[Tensor, "batch"]]] = None,  # noqa: F821
    # predict_fn: Optional[
    #     Callable[[Float[Tensor, "batch n_ctx d_vocab_out"]], S]  # noqa: F722
    # ] = None,
    # acc_fn: Optional[Callable[[S, S], Bool[Tensor, "batch"]]] = (  # noqa: F821
    #     lambda x, y: x == y
    # ),
    # example_log: Optional[list[Integer[Tensor, "batch n_ctx"]]] = None,
    # example_results_log: Optional[list[Tuple[int, int]]] = None,
    generate_data_fn: Callable[
        [int, int, int], Integer[Tensor, "batch n_ctx"]  # noqa: F722
    ],
    device: str | torch.device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    ),
    max_examples_to_log: Optional[int] = 100,
    seed: int = 0,
    cache: bool = False,
    # baseline_time: Optional[int] = None,
):
    """Creates a compression tool for model evaluation and comparison.

    This tool is designed to evaluate and compare models by running specified code snippets
    that define model matrices, synthetic model matrices, and computations from these matrices.
    It assesses the efficiency and accuracy of synthetic model representations.

    Args:
        model_path (str | Path): Path to the model file.
        timeout (int | None, optional): Maximum time allowed for code execution. Defaults to None.
        user (str | None, optional): Identifier for the user executing the tool. Defaults to None.
        exec_file_remote (str, optional): Remote path for executing files. Defaults to "/tmp/exec.py".
        session_id (Optional[str], optional): Unique session identifier. Defaults to None.
        safe_sandbox_name (Optional[str], optional): Name of the safe sandbox environment. Defaults to "without_model".
        half_safe_sandbox_name (Optional[str], optional): Name of the half-safe sandbox environment. Defaults to "without_model2".
        reduce_fn (Callable, optional): Function to reduce model outputs. Defaults to None.
        compare_fn (Optional[Callable[[S, S], Float[Tensor, "batch"]]], optional): Function to compare model outputs. Defaults to None.
        generate_data_fn (Callable[[int, int, int], Integer[Tensor, "batch n_ctx"]]): Function to generate data for model evaluation, given n_ctx, d_vocab, and a random seed.
        max_examples_to_log (Optional[int], optional): Maximum number of examples to log. Defaults to 100.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.
        cache (bool, optional): Whether to cache results. Defaults to False.
    """

    model_funcs = ModelInterventionFunctions(
        model_path=model_path,
        # label_fn=label_fn,
        predict_fn=reduce_fn,
        # acc_fn=acc_fn,
        device=device,
    )

    if compare_fn is None:
        compare_fn = torch.nn.functional.cross_entropy

    @torch.no_grad()
    async def run(
        model_matrices_code: str,
        model_matrices_function_name: str,
        synthetic_model_matrices_code: str,
        synthetic_model_matrices_function_name: str,
        compute_model_from_matrices_code: str,
        compute_model_from_matrices_function_name: str,
    ) -> ToolResult:
        """Run the code to define the variable or argumentless {function_name}, and then uses the output of {function_name}.
        model_matrices_code is run to define {model_matrices_function_name}, *with access to the model and state from previous executions*, which should return a dictionary of matrices or paths of the particular model of interest.
        synthetic_model_matrices_code is run to define {synthetic_model_matrices_function_name}, *without access to the model*, which should return a dictionary of matrices that are being computed synthetically.
        compute_model_from_matrices_code is run to define {compute_model_from_matrices_function_name}, which should take as its first argument a batch of data (Tensor of shape (batch, n_ctx)) and the matrices from the model, some of which may be replaced by matrices from the synthetic model; these are passed as keyword arguments.

        Note that this tool MUST be called to recieve a nonzero score, as this tool registers the attempts for scoring.

        Grading is done on the basis of how much code is required to compute the synthetic model matrices (less is better), how many parameters remain in the model matrices that are not replaced by synthetic matrices (fewer is better), the matrix 2-norm of the matrices that remain from the original model matrices (smaller is better), and the average cross-entropy-loss between the softmaxed logits of the model and the softmaxed logits of the reconstructed model (non-synthetic) (smaller is better), and the average cross-entropy-loss between the softmaxed logits of the reconstructed model and the softmaxed logits of the synthetic model (smaller is better).

        Args:
            model_matrices_code (str): Initial code to run to define {model_matrices_function_name}, *with access to the model and state from previous executions*
            model_matrices_function_name: a variable or function that should return a dictionary of matrices or paths of the particular model of interest
            synthetic_model_matrices_code (str): Initial code to run to define {synthetic_model_matrices_function_name}, *without access to the model*
            synthetic_model_matrices_function_name: a variable or function that should return a dictionary of matrices that are being computed synthetically.  The keys must be a subset of the keys returned by model_matrices_function_name.
            compute_model_from_matrices_code (str): Initial code to run to define {compute_model_from_matrices_function_name}
            compute_model_from_matrices_function_name: a function that should take as its first argument a batch of data (Tensor of shape (batch, n_ctx)) and the matrices from the model, some of which may be replaced by matrices from the synthetic model, as keyword arguments.  It should return a batch of logits in the same shape as the model output.

        Returns:
            ToolResult: Messages that came up during execution, and finally a string "mean CE(model, model_matrices), mean CE(model_matrices, synthetic_model_matrices), # matrices total, # matrices remaining, 2-norm of model matrices, 2-norm of remaining model matrices, # parameters total, # parameters remaining, length of code used to define synthetic_model_matrices (excluding whitespace and comments)
        """
        nonlocal seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        seed = np.random.randint(2**32)

        data = generate_data_fn(
            model_funcs.model.cfg.n_ctx, model_funcs.model.cfg.d_vocab, seed
        )
        assert isinstance(
            data, torch.Tensor
        ), f"generate_data_fn must return a Tensor, not {type(data)}"
        assert (
            data.ndim == 2
        ), f"generate_data_fn must return a 2D tensor, not a tensor of shape {data.shape}"

        # get model matrices
        model_matrices, messages = await python_exec_code_func_in_sandbox(
            model_matrices_code,
            model_matrices_function_name,
            timeout=timeout,
            user=user,
            exec_file_remote=exec_file_remote,
            session_id=session_id,
            cache=cache,
        )
        messages.insert(
            0,
            ContentText(
                text=f"Model Matrices Code:\n```python\n{model_matrices_code}\n```"
            ),
        )
        model_matrices, messages = sanitize_model_matrices(
            model_matrices, messages, descr="model matrices"
        )

        synthetic_model_matrices, extra_messages = (
            await python_exec_code_func_in_sandbox(
                synthetic_model_matrices_code,
                synthetic_model_matrices_function_name,
                timeout=timeout,
                user=user,
                exec_file_remote=exec_file_remote,
                session_id=session_id,
                cache=cache,
                name=safe_sandbox_name,
            )
        )
        extra_messages.insert(
            0,
            ContentText(
                text=f"Synthetic Model Matrices Code:\n```python\n{model_matrices_code}\n```"
            ),
        )
        synthetic_model_matrices, extra_messages = sanitize_model_matrices(
            model_matrices, extra_messages, descr="synthetic model matrices"
        )
        messages += extra_messages

        extra_keys = set(synthetic_model_matrices) - set(model_matrices)
        if extra_keys:
            messages.append(
                ContentText(
                    text=f"Synthetic model matrices has keys not in model matrices: {extra_keys}"
                )
            )
            for k in extra_keys:
                del synthetic_model_matrices[k]

        compute_model_from_matrices_code = (
            f"import torch\n{compute_model_from_matrices_code}"
        )
        full_synthetic_model_matrices = model_matrices | synthetic_model_matrices
        model_matrices_str = (
            "{"
            + ", ".join(
                f"{k!r}: torch.tensor({v.tolist()!r})"
                for k, v in model_matrices.items()
            )
            + "}"
        )
        synthetic_model_matrices_str = (
            "{"
            + ", ".join(
                f"{k!r}: torch.tensor({v.tolist()!r})"
                for k, v in synthetic_model_matrices.items()
            )
            + "}"
        )
        full_synthetic_model_matrices_str = (
            "{"
            + ", ".join(
                f"{k!r}: torch.tensor({v.tolist()!r})"
                for k, v in full_synthetic_model_matrices.items()
            )
            + "}"
        )
        data_str = f"torch.tensor({data.tolist()!r})"
        compute_real_model_from_matrices_function_name = f"{compute_model_from_matrices_function_name}({data_str}, **{model_matrices_str})"
        compute_synthetic_model_from_matrices_function_name = f"{compute_model_from_matrices_function_name}({data_str}, **{full_synthetic_model_matrices_str})"

        model_matrices_results, extra_messages = await python_exec_code_func_in_sandbox(
            compute_model_from_matrices_code,
            compute_real_model_from_matrices_function_name,
            timeout=timeout,
            user=user,
            exec_file_remote=exec_file_remote,
            session_id=session_id,
            cache=cache,
            name=half_safe_sandbox_name,
        )
        extra_messages.insert(
            0,
            ContentText(
                text=f"Compute Model From Matrices Code:\n```python\n{compute_model_from_matrices_code}\n```\n```python\n{compute_real_model_from_matrices_function_name}\n```"
            ),
        )
        messages += extra_messages
        try:
            model_matrices_results = tensorify(model_matrices_results)
        except (ValueError, TypeError) as e:
            raise ToolError(
                f"Model matrices results must be a Tensor, not {type(model_matrices_results)} ({e})\nMessages: {messages}"
            )

        synthetic_model_matrices_results, extra_messages = (
            await python_exec_code_func_in_sandbox(
                compute_model_from_matrices_code,
                compute_synthetic_model_from_matrices_function_name,
                timeout=timeout,
                user=user,
                exec_file_remote=exec_file_remote,
                session_id=session_id,
                cache=cache,
                name=safe_sandbox_name,
            )
        )
        extra_messages.insert(
            0,
            ContentText(
                text=f"Compute Model From Matrices Code:\n```python\n{compute_model_from_matrices_code}\n```\n```python\n{compute_synthetic_model_from_matrices_function_name}\n```"
            ),
        )
        messages += extra_messages
        try:
            synthetic_model_matrices_results = tensorify(
                synthetic_model_matrices_results
            )
        except (ValueError, TypeError) as e:
            raise ToolError(
                f"Synthetic model matrices results must be a Tensor, not {type(synthetic_model_matrices_results)} ({e})\nMessages: {messages}"
            )

        logger.debug("Running model on %s samples", data.shape)
        true_results = await asyncio.to_thread(model_funcs.predict_data, data)

        if (
            true_results.shape != synthetic_model_matrices_results.shape
            and reduce_fn is not None
        ):
            synthetic_model_matrices_results = reduce_fn(
                synthetic_model_matrices_results
            )

        try:
            ce_model_model_matrices = compare_fn(true_results, model_matrices_results)
            ce_model_matrices_synthetic = compare_fn(
                model_matrices_results, synthetic_model_matrices_results
            )
        except Exception as e:
            raise ToolError(f"Error comparing results: {e}\nMessages: {messages}")

        getsource_recursively_source = getsource_recursively(
            getsource_recursively,
            include_comments=False,
            include_type_annotations=False,
            import_modules=("typing", "inspect", "collections"),
        )
        synthetic_full_code, extra_messages = await python_exec_code_func_in_sandbox(
            getsource_recursively_source,
            f"getsource_recursively({synthetic_model_matrices_function_name})",
            timeout=timeout,
            user=user,
            exec_file_remote=exec_file_remote,
            session_id=session_id,
            cache=cache,
            name=safe_sandbox_name,
        )
        messages += extra_messages
        synthetic_code_length = (
            None
            if synthetic_full_code is None
            else len(synthetic_full_code.replace(" ", "").replace("\n", ""))
        )

        messages.append(
            ContentText(
                text=f"""CE(model, model_matrices): {ce_model_model_matrices.mean().item()}
CE(model_matrices, synthetic_model_matrices): {ce_model_matrices_synthetic.mean().item()}
# matrices total: {len(model_matrices)}
# matrices remaining: {len(synthetic_model_matrices)}
2-norm of model matrices: {sum(v.norm()**2 for v in model_matrices.values()).sqrt().item()}
2-norm of remaining model matrices: {sum(v.norm()**2 for v in synthetic_model_matrices.values()).sqrt().item()}
# parameters total: {sum(v.numel() for v in model_matrices.values())}
# parameters remaining: {sum(v.numel() for v in synthetic_model_matrices.values())}
length of code used to define synthetic_model_matrices: {synthetic_code_length}"""
            )
        )

        save_compression_attempt(
            **{
                "synthetic_model_matrices": synthetic_model_matrices.detach()
                .cpu()
                .numpy(),
                "model_matrices": model_matrices.detach().cpu().numpy(),
                "synthetic_model_matrices_code": synthetic_model_matrices_code,
                "model_matrices_code": model_matrices_code,
                "compute_model_from_matrices_code": compute_model_from_matrices_code,
                "synthetic_model_matrices_function_name": synthetic_model_matrices_function_name,
                "model_matrices_function_name": model_matrices_function_name,
                "compute_model_from_matrices_function_name": compute_model_from_matrices_function_name,
                "ce_model_model_matrices": ce_model_model_matrices.mean().item(),
                "ce_model_matrices_synthetic": ce_model_matrices_synthetic.mean().item(),
                "synthetic_full_code": synthetic_full_code,
                "data": data.detach().cpu().numpy(),
            }
        )

        return messages

    return run
