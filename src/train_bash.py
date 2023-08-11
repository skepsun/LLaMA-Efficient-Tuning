# from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
# replace_llama_attn_with_flash_attn()
from llmtuner import run_exp


def main():
    run_exp()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
