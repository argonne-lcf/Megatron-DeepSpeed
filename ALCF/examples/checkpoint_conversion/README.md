# Converting `AutoModel` to DeepSpeed ZeRO Checkpoint

We would like to convert an (arbitrarily large) HuggingFace model to a ZeRO
checkpoint so that we can use it for continual pre-training with
Megatron-DeepSpeed.

Previously, we had been using the approach from [ALCF / examples /
finetune_llama3](/ALCF/examples/finetune_llama3/README.md).

In particular, this approach works by:

1. Instantiate the Megatron-DeepSpeed (MDS) model as normal (with empty
   weights), from [\[here\]](/tools/hf2megads_weight_converter.py#L712)

      ```python
      from megatron.model import GPTModelPipe
      ds_model = GPTModelPipe(config, num_tokentypes=0, parallel_output=True)
      ```

1. Instantiate the HF model \[[here\]](/tools/hf2megads_weight_converter.py#L725)

    ```python
    from transformers import AutoModel
    hf_model = AutoModel.from_pretrained("meta-llama/llama-3.3-70b-instruct")
    ```

3. Instantiate optimizer [\[here\]](/tools/hf2megads_weight_converter.py#L736)

1. Layer by layer, copy the weights from the HF model to the MDS model
   \[[here\]](/tools/hf2megads_weight_converter.py#L766)


Unfortunately, for very large models, this will slowly consume available host
memory until it is exhausted causing the application to crash.

## Proposed Solution

Our proposed solution is simple and entirely contained in [ALCF / examples / checkpoint_conversion / hf_to_zero.py](/ALCF/examples/checkpoint_conversion/hf_to_zero.py).

Explicitly:

1. Create the HF model as normal
2. Pass it to `deepspeed.initalize(...)` to create the `DeepSpeedEngine`
3. `DeepSpeedEngine.save_checkpoint(...)` to save the checkpoint.


To run:

```bash
launch python3 \
  ALCF/examples/checkpoint_conversion/hf_to_zero.py \
  --zero-stage=3 \
  --device=cpu \
  --model='meta-llama/llama-3.3-70b-instruct'
```

> [!WARNING]
> I believe this approach is still not finished because I expect there will be
> naming mismatches between the layers of the HF model (now saved in our ZeRO
> checkpoint) and what our MDS model expects.
> 
> This requires further testing to confirm, but we are now able to successfully
> convert the 70B model to a ZeRO checkpoint.

## Estimate Memory Needs for Llama-3.3-70B-Instruct

Deepspeed provides a nice mechanism for determining the memory needs of a model.

We provide below the summary for the Llama-3.3-70B-Instruct model of interest.

|       Model Name       | Model Size | Model Parameters | Largest Layer Parameters | Memory Needed |
|:----------------------:|:----------:|:----------------:|:------------------------:|:-------------:|
| Llama-3.3-70B-Instruct |     70B    |      69503M      |           1050M          |    70.45GB   | 



```bash
$ python3 -c 'from transformers import AutoModel; \
∙ from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; \
∙ model = AutoModel.from_pretrained("meta-llama/Llama-3.3-70B-Instruct"); \
∙ estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=12, num_nodes=4)'
```

<details closed><summary>Output</summary>


```bash
Loading checkpoint shards: 100%|████████████████| 30/30 [08:28<00:00, 16.94s/it]
Estimated memory needed for params, optim states and gradients for a:
HW: Setup with 4 nodes, 12 GPUs per node.
SW: Model with 69503M total params, 1050M largest layer params.
  per CPU  |  per GPU |   Options
  436.93GB |   3.91GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
 4660.54GB |   3.91GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
  388.38GB |   6.61GB | offload_param=none, offload_optimizer=cpu , zero_init=1
 4660.54GB |   6.61GB | offload_param=none, offload_optimizer=cpu , zero_init=0
   70.45GB |  28.19GB | offload_param=none, offload_optimizer=none, zero_init=1
 4660.54GB |  28.19GB | offload_param=none, offload_optimizer=none, zero_init=0
took: 0h:08m:44s
```

</details>


- Model States and Memory Needs for Llama-3.3-70B-Instruct:


    |  per CPU  | per GPU |                         Options                         |
    |:---------:|:-------:|:-------------------------------------------------------:|
    |  436.93GB |  3.91GB |  offload_param=cpu, offload_optimizer=cpu, zero_init=1  |
    | 4660.54GB |  3.91GB |  offload_param=cpu, offload_optimizer=cpu, zero_init=0  |
    |  388.38GB |  6.61GB |  offload_param=none, offload_optimizer=cpu, zero_init=1 |
    | 4660.54GB |  6.61GB |  offload_param=none, offload_optimizer=cpu, zero_init=0 |
    |  70.45GB  | 28.19GB | offload_param=none, offload_optimizer=none, zero_init=1 |
    | 4660.54GB | 28.19GB | offload_param=none, offload_optimizer=none, zero_init=0 |



