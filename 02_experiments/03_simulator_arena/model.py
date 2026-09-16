from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch


def load_role_vectors(vector_root: str | Path, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    root = Path(vector_root) / f"layer_{layer}"
    return (
        torch.load(root / "user.pt", map_location="cpu", weights_only=True).float(),
        torch.load(root / "assistant.pt", map_location="cpu", weights_only=True).float(),
    )


class Qwen35:
    """Qwen 3.5 user simulation with role-vector steering."""

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
        vector_root: str | Path | None = None,
        layer: int = 11,
        alpha: float = 0.0,
    ) -> str:
        self.load()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        prompt_tokens = self.model.to_tokens(self.render(messages), prepend_bos=False).to(self.model_device)

        context = nullcontext()
        if alpha != 0:
            user, assistant = load_role_vectors(vector_root, layer)
            direction = user - assistant
            direction /= torch.linalg.vector_norm(direction).clamp_min(1e-12)

            def edit(activation, hook=None):
                hidden = activation.float()
                norms = torch.linalg.vector_norm(hidden, dim=-1, keepdim=True).clamp_min(1e-12)
                offset = alpha * norms * direction.to(activation.device).view(1, 1, -1)
                return (hidden + offset).to(activation.dtype)

            context = self.model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_out", edit)])

        with torch.inference_mode(), context:
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
