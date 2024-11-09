# Converting Checkpoints

## Megatron $\rightarrow$ 🤗 HuggingFace

On Aurora,

- Setup:

    ```bash
    CKPT_ROOT="/flare/Aurora_deployment/AuroraGPT-Testing/foremans/rollback-41k8/Megatron-DeepSpeed-41800/checkpoints/ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb3072_sp1_pp1_tp1_bf16_optadamw_lr0.00020_lwf0.05";

    LAST_STEP=$(cat "${CKPT_ROOT}/latest_checkpointed_iteration.txt")
    GLOBAL_STEP="${GLOBAL_STEP:-${LAST_STEP}}"

    SRC="${CKPT_ROOT}/global_step${GLOBAL_STEP}"

    OUTPUT_PARENT="/flare/Aurora_deployment/AuroraGPT-Checkpoints/production-checkpoints/aGPT-7B/HF"
    DST="${OUTPUT_PARENT}/global_step${GLOBAL_STEP}_hf"

    printf "SRC: %s\n DST: %s\n" "${SRC}" "${DST}"
    ```

- Convert:

    ```bash
    python3 Megatron-DeepSpeed/mds_to_hf.py \
        --mds_checkpoint "${SRC}/mp_rank_00_model_states.pt" \
        --output_dir "${DST}" \
        --cache_dir "./.cache"
    ```

<!-- ```bash -->
<!-- # [SRC]: Megatron-DeepSpeed checkpoint -->
<!-- GLOBAL_STEP=77000 -->
<!-- CKPT_ROOT="/flare/Aurora_deployment/AuroraGPT-Testing/foremans/rollback-41k8/Megatron-DeepSpeed-41800/checkpoints/ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb3072_sp1_pp1_tp1_bf16_optadamw_lr0.00020_lwf0.05" -->
<!-- SRC="${CKPT_ROOT}/global_step${GLOBAL_STEP}" -->
<!-- # [DST]: HuggingFace checkpoint -->
<!-- OUTPUT_PARENT="/flare/Aurora_deployment/AuroraGPT-Checkpoints/production-checkpoints/aGPT-7B/HF/" -->
<!-- DST="${OUTPUT_PARENT}/global_step${GLOBAL_STEP}_hf" -->
<!-- # Convert [SRC] --> [DST] -->
<!-- # using `argonne-lcf/Megatron-DeepSpeed/mds_to_hf.py` -->
<!-- # see: -->
<!-- #   https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/main/mds_to_hf.py -->
<!-- # for implementation -->
<!-- python3 \ -->
<!--     Megatron-DeepSpeed/mds_to_hf.py \ -->
<!--     --mds_checkpoint "${SRC}/mp_rank_00_model_states.pt" \ -->
<!--     --output_dir "${DST}" \ -->
<!--     --cache_dir "./.cache" -->
<!-- ``` -->

## Use in 🤗 `transformers`

```python
from pathlib import Path
import time
from rich import print
from typing import Optional
from transformers import LlamaForCausalLM, AutoTokenizer

def load_model(ckpt_dir: str, step: Optional[int] = None):
    if step is None:
        fp = Path(ckpt_dir)
    else:
        fp = Path(ckpt_dir).joinpath(f"global_step{step}_hf")
    print(f"Loading ckpt from: {fp}")
    if fp.exists():
        model = LlamaForCausalLM.from_pretrained(fp.as_posix())
        print(f"{model=}")
        return model

    raise FileNotFoundError(f"Unable to locate checkpoint at: {fp}")


def eval_model(
        model: torch.nn.Module,
        max_length: int = 64,
        prompt: Optional[str] = None,
        tokenizer: Optional[AutoTokenizer] = None,
) -> str:
    prompt = "What is it like in there?" if prompt is None else prompt
    tokenizer = (
        AutoTokenizer.from_pretrained("meta-llama/Llama-2-7B-hf")
        if tokenizer is None else tokenizer
    )
    output = (
        tokenizer.batch_decode(
            model.generate(
                **tokenizer(prompt, return_tensors="pt"),
                 max_length=max_length,
            ),
            clean_up_tokenization_spaces=True,
            skip_special_tokens=True,
        )[0]
    )
    return output


def loop_over_checkpoints(
    steps_list: list[int],
    ckpt_dir: str,
    max_length: int = 128,
    prompt: Optional[str] = None,
):
    for step in steps_list:
        t0 = time.perf_counter()
        prompt = "What is it like in there?" if prompt is None else prompt
        print(f"\n Loading model from checkpoint at global step: {step}")
        outputs = eval_model(
            load_model(step, ckpt_dir),
            max_length=max_length,
            prompt=prompt,
        )
        print(f"{outputs}")
        print(f"\ntook: {time.perf_counter() - t0:.6f}s\n")
```

```python
>>> ckpt_dir = "/flare/Aurora_deployment/AuroraGPT-Checkpoints/production-checkpoints/aGPT-7B/HF/"
>>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7B-hf")
>>> model = load_model(76300, ckpt_dir)
Loading ckpt from:
/flare/Aurora_deployment/AuroraGPT-Checkpoints/production-checkpoints/aGPT-7B/HF/global_step76300_hf
model=LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((4096,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)

>>> print(
...     eval_model(
...         model,
...         max_length=128,
...         prompt="What is it like in there?",
...         tokenizer=tokenizer
...     )
... )
Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, whereit will always be FP32)
What is it like in there?
I've been in there a few times. It's a pretty cool place.
I've been in there a few times. It's a pretty cool place.
I've been in there a few times. It's a pretty cool place.
I've been in there a few times. It's a pretty cool place.
I've been in there a few times. It's a pretty cool place.
I've been in there a few times. It's a pretty cool place.
I've been in
```

## Helper Script

```bash
convert_mds_to_hf() {
    if [[ "$#" -eq 3 ]]; then
        GLOBAL_STEP=$1
        CKPT_ROOT=$2
        OUTPUT_PARENT=$3
    elif [[ "$#" -eq 2 ]]; then
        GLOBAL_STEP=$1
        CKPT_ROOT=$2
        OUPUT_PARENT=$(pwd)
    elif [[ "$#" -eq 1 ]]; then
        GLOBAL_STEP=$1
        CKPT_ROOT="/flare/Aurora_deployment/AuroraGPT-Testing/foremans/rollback-41k8/Megatron-DeepSpeed-41800/checkpoints/ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb3072_sp1_pp1_tp1_bf16_optadamw_lr0.00020_lwf0.05/";
        OUPUT_PARENT=$(pwd)
    else
        echo "Expected exactly 1, 2, or 3 arguments (global_step, src, dst, respectively)"
        exit
    fi
    SRC="${CKPT_ROOT}/global_step${GLOBAL_STEP}"
    DST="${OUTPUT_PARENT}/global_step${GLOBAL_STEP}_hf"
    if [[ -d "${SRC}" ]]; then
        echo "Converting checkpoint @ global step ${GLOBAL_STEP}"
        echo "\tsrc = ${SRC}\n"
        echo "\tdst = ${DST}\n"
        python3 mds_to_hf.py \
            --mds_checkpoint "${SRC}/mp_rank_00_model_states.pt" \
            --output_dir "${DST}" \
            --cache_dir "./.cache"
    else
        echo "Unable to locate directory ${SRC}. Exiting"
        exit 1
    fi
}
```
