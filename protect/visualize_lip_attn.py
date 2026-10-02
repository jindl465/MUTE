"""
Gradient capture for lip cross-attention target construction.

LipAttnGradCapture records the lip audio cross-attention map A and estimates
first- and second-order spatial sensitivities with respect to the query input.
The attack then converts A, A', and A'' into the paper target distribution.
"""

import torch

from hallo.models.attention import AudioTemporalBasicTransformerBlock


class _WeightCaptureProcessor:
    """Attention processor that supports double backpropagation."""

    def __init__(self):
        self.q_input: torch.Tensor | None = None
        self.attn_weights: torch.Tensor | None = None

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs,
    ):
        self.q_input = hidden_states
        kv_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        query = attn.head_to_batch_dim(attn.to_q(hidden_states))
        key = attn.head_to_batch_dim(attn.to_k(kv_states))
        value = attn.head_to_batch_dim(attn.to_v(kv_states))

        scores = torch.bmm(query, key.transpose(-1, -2)) * (query.shape[-1] ** -0.5)
        if attention_mask is not None:
            scores = scores + attention_mask

        weights = torch.softmax(scores, dim=-1)
        self.attn_weights = weights

        hidden_states = torch.bmm(weights, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class LipAttnGradCapture:
    """Install processors on attn2_2 layers and compute A, A', and A''."""

    def __init__(self, denoising_unet: torch.nn.Module):
        self._processors: list[tuple] = []
        self._hooks: list = []

        for module in denoising_unet.modules():
            if not isinstance(module, AudioTemporalBasicTransformerBlock):
                continue
            attn_module = getattr(module, "attn2_2", None)
            if attn_module is None:
                continue

            processor = _WeightCaptureProcessor()
            original_processor = attn_module.processor
            attn_module.set_processor(processor)
            self._processors.append((attn_module, original_processor, processor))
            self._hooks.append(attn_module.register_forward_pre_hook(self._query_pre_hook))

        if not self._processors:
            raise RuntimeError(
                "No attn2_2 layers were found. Disable xformers before target capture."
            )
        print(f"[LipAttnGradCapture] Hooked {len(self._processors)} attn2_2 layer(s).")

    def _query_pre_hook(self, module, args):
        query_input = args[0].detach().requires_grad_(True)
        for attn_module, _, processor in self._processors:
            if attn_module is module:
                processor.q_input = query_input
                break
        return (query_input,) + args[1:]

    def clear(self) -> None:
        for _, _, processor in self._processors:
            processor.q_input = None
            processor.attn_weights = None

    def remove(self) -> None:
        for attn_module, original_processor, _ in self._processors:
            attn_module.set_processor(original_processor)
        self._processors.clear()
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def compute_all(self) -> list[dict[str, torch.Tensor]]:
        results = []
        for _, _, processor in self._processors:
            if processor.attn_weights is None or processor.q_input is None:
                continue

            q_input = processor.q_input
            weights = processor.attn_weights
            batch = q_input.shape[0]
            heads = weights.shape[0] // batch

            A = (
                weights.reshape(batch, heads, weights.shape[1], weights.shape[2])
                .max(dim=3).values
                .mean(dim=1)
            )

            try:
                first_grad = torch.autograd.grad(
                    A.sum(),
                    q_input,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                Ap = first_grad.abs().mean(dim=-1)
            except RuntimeError as exc:
                print(f"[warn] first-order attention gradient failed: {exc}")
                continue

            try:
                second_grad = torch.autograd.grad(
                    first_grad.sum(),
                    q_input,
                    retain_graph=False,
                )[0]
                App = second_grad.abs().mean(dim=-1)
            except RuntimeError as exc:
                print(f"[warn] second-order attention gradient failed: {exc}")
                App = torch.zeros_like(Ap)

            S = (Ap + 0.5 * App) / A.clamp_min(1e-8)
            results.append({
                "A": A.detach(),
                "Ap": Ap.detach(),
                "App": App.detach(),
                "S": S.detach(),
            })
        return results
