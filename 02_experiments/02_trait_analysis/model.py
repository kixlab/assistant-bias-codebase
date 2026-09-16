from __future__ import annotations

from pathlib import Path

import torch


def load_role_vectors(vector_root: str | Path, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    root = Path(vector_root) / f"layer_{layer}"
    return (
        torch.load(root / "user.pt", map_location="cpu", weights_only=True).float(),
        torch.load(root / "assistant.pt", map_location="cpu", weights_only=True).float(),
    )


class Qwen35:
    """Qwen 3.5 trait response generation and activation replay."""

    def __init__(self, *, device: str = "cuda") -> None:
        self.device = device
        self.model = None

    def load(self) -> None:
        if self.model is not None:
            return
        from transformer_lens.model_bridge import TransformerBridge

        self.model = TransformerBridge.boot_transformers(
            "Qwen/Qwen3.5-9B",
            device=self.device,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model.eval()
        self.tokenizer = self.model.tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_device = next(self.model.parameters()).device

    def render(self, messages: list[dict], *, add_generation_prompt: bool = True) -> str:
        self.load()
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )

    def generate(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int = 1024,
        seed: int = 42,
    ) -> str:
        self.load()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        prompt_tokens = self.model.to_tokens(self.render(messages), prepend_bos=False).to(self.model_device)
        with torch.inference_mode():
            output = self.model.generate(
                prompt_tokens,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                stop_at_eos=True,
                use_past_kv_cache=True,
                verbose=False,
            )
        response = output[0, prompt_tokens.shape[1] :].detach().cpu()
        return self.tokenizer.decode(response.tolist(), skip_special_tokens=True).strip()

    def response_hidden(
        self,
        messages: list[dict],
        response: str,
        *,
        layers: list[int],
        pooling: str = "first_token",
    ) -> dict[int, torch.Tensor]:
        self.load()
        prompt_ids = self.tokenizer.encode(self.render(messages), add_special_tokens=False)
        full_ids = self.tokenizer.encode(
            self.render([*messages, {"role": "assistant", "content": response}], add_generation_prompt=False),
            add_special_tokens=False,
        )
        start, end = len(prompt_ids), len(full_ids)
        while end > start and full_ids[end - 1] in self.tokenizer.all_special_ids:
            end -= 1
        if end <= start:
            raise ValueError("Response has no content tokens")
        indices = [start] if pooling == "first_token" else list(range(start, end))
        names = {layer: f"blocks.{layer}.hook_out" for layer in layers}
        name_set = set(names.values())
        tokens = torch.tensor([full_ids], dtype=torch.long, device=self.model_device)
        with torch.inference_mode():
            _, cache = self.model.run_with_cache(
                tokens,
                names_filter=lambda name: name in name_set,
                return_type=None,
            )
        return {
            layer: cache[name][0, indices].detach().float().cpu().mean(dim=0)
            for layer, name in names.items()
        }
