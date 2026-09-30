"""Llama-3.2-1B-Instruct with attn_implementation="flash_lab" against the same model on SDPA.

Needs transformers and the checkpoint in the local Hugging Face cache (scripts/pace/models.sh);
skipped otherwise. The logit check mirrors the kernel tests: against an fp32 copy of the model,
flash_lab's bf16 logits may be at most twice as far off as SDPA's.

Greedy generation follows SDPA's tokens until SDPA's own top two logits are within two bf16
steps of each other; at such a near-tie any change in rounding can pick the other token. On
these prompts PyTorch's eager attention leaves SDPA at the same near-ties, so exact agreement on
every prompt is not a fair requirement.
"""

import math

import pytest
import torch

transformers = pytest.importorskip("transformers")

import flash_lab.hf  # noqa: E402

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
NEW_TOKENS = 64
PROMPTS = [
    "Explain why the KV cache grows linearly with the sequence length.",
    "Write a haiku about GPUs.",
    "What is the capital of France? Answer in one word.",
    "List three prime numbers greater than 100.",
    "Summarize the idea of online softmax in two sentences.",
]

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


@pytest.fixture(scope="module")
def checkpoint():
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(MODEL, local_files_only=True)
    except Exception:
        pytest.skip(f"{MODEL} is not in the local Hugging Face cache")


@pytest.fixture(scope="module")
def model(checkpoint):
    flash_lab.hf.register()
    return transformers.AutoModelForCausalLM.from_pretrained(
        checkpoint, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda"
    ).eval()


@pytest.fixture(scope="module")
def prompts(checkpoint):
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True, return_tensors="pt"
        ).cuda()
        for p in PROMPTS
    ]


def greedy(model, input_ids, impl, cache=None):
    """Greedy tokens and the logits of every step; a static cache also compiles the decode step,
    with fullgraph=True so that any graph break raises."""
    model.set_attn_implementation(impl)
    compile_kwargs = {}
    if cache is not None:
        compile_kwargs = dict(
            past_key_values=cache, compile_config=transformers.CompileConfig(fullgraph=True)
        )
    out = model.generate(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=NEW_TOKENS,
        min_new_tokens=NEW_TOKENS,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=model.generation_config.eos_token_id[0],
        return_dict_in_generate=True,
        output_logits=True,
        **compile_kwargs,
    )
    return out.sequences[0, input_ids.shape[1] :], torch.stack(out.logits)[:, 0]


@pytest.fixture(scope="module")
def sdpa_runs(model, prompts):
    return [greedy(model, p, "sdpa") for p in prompts]


def assert_follows(run, reference) -> int | None:
    """The same tokens as the reference, up to a step where the reference's top two logits were
    within two bf16 steps. Returns that step, or None if all tokens match."""
    tokens, ref_tokens, ref_logits = run[0], reference[0], reference[1]
    differ = (tokens != ref_tokens).nonzero()
    if len(differ) == 0:
        return None
    step = int(differ[0])
    first, second = ref_logits[step].topk(2).values.tolist()
    tie = 2 * torch.finfo(torch.bfloat16).eps * 2 ** math.floor(math.log2(abs(first)))
    assert first - second <= tie, f"left the reference at token {step}, margin {first - second}"
    return step


@torch.no_grad()
def test_logit_error_at_most_twice_sdpa(checkpoint, model):
    ids = torch.randint(0, 128000, (2, 300), generator=torch.Generator().manual_seed(0)).cuda()
    reference = transformers.AutoModelForCausalLM.from_pretrained(
        checkpoint, dtype=torch.float32, attn_implementation="sdpa", device_map="cuda"
    ).eval()
    want = reference(ids).logits.double()
    del reference
    errors = {}
    for impl in ("sdpa", "flash_lab"):
        model.set_attn_implementation(impl)
        errors[impl] = (model(ids).logits.double() - want).abs().max().item()
    assert errors["flash_lab"] <= 2 * errors["sdpa"], errors


def test_greedy_generation_follows_sdpa(model, prompts, sdpa_runs):
    for p, reference in zip(prompts, sdpa_runs, strict=True):
        assert_follows(greedy(model, p, "flash_lab"), reference)


def test_static_cache_decode_compiles_in_one_graph(model, prompts, sdpa_runs):
    # The decode step runs compiled (CUDA graphs) on the static cache buffers. One cache size
    # for all prompts keeps it to a single compilation.
    torch._dynamo.reset()
    cache = transformers.StaticCache(config=model.config, max_cache_len=256)
    for p, reference in zip(prompts, sdpa_runs, strict=True):
        cache.reset()
        assert_follows(greedy(model, p, "flash_lab", cache), reference)
