"""Small compatibility wrapper around RL4CO's MDAM decoder.

RL4CO 0.6.0 computes the embedding-glimpse branch in MDAM, but it (a) never
sets ``is_vrp`` and (b) does not store the first mask before using it in
``mask ^ mask_first``.  The original MDAM paper uses ``n_EG=2`` for TSP, so
the default implementation crashes as soon as the glimpse branch is reached.
This module keeps the same architecture/parameter names and only fixes those
two implementation issues.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rl4co.models.zoo.mdam.decoder import MDAMDecoder
from rl4co.models.zoo.mdam.policy import MDAMPolicy
from rl4co.utils.decoding import decode_logprobs, get_log_likelihood


class FixedMDAMDecoder(MDAMDecoder):
    def __init__(self, *args, env_name: str = "tsp", **kwargs):
        super().__init__(*args, env_name=env_name, **kwargs)
        self.is_vrp = env_name in ("cvrp", "sdvrp")
        self.is_tsp = env_name == "tsp"

    def _get_logprobs(self, fixed, td, path_index, normalize=True):
        # RL4CO's MDAM returns masked logits here.  Decoding and the
        # REINFORCE log-likelihood expect log-probabilities, so normalize.
        logits, mask = super()._get_logprobs(
            fixed, td, path_index, normalize=normalize
        )
        return F.log_softmax(logits, dim=-1), mask

    def forward(
        self,
        td,
        encoded_inputs,
        env,
        attn,
        V,
        h_old,
        encoder,
        **decoder_kwargs,
    ):
        output_list = []
        td_list = [td.clone() for _ in range(self.num_paths)]
        for i in range(self.num_paths):
            _encoded_inputs = encoded_inputs.clone()
            fixed = self._precompute(_encoded_inputs, path_index=i)
            logprobs, _ = self._get_logprobs(fixed, td_list[i], i)
            output_list.append(logprobs[:, 0, :])
            output_list[-1] = torch.max(
                output_list[-1],
                torch.full_like(output_list[-1], -1e9),
            )

        loss_kl_divergence = output_list[0].new_zeros(())
        if self.num_paths > 1:
            kl_divergences = []
            for _i in range(self.num_paths):
                for _j in range(self.num_paths):
                    if _i == _j:
                        continue
                    divergence = torch.sum(
                        torch.exp(output_list[_i])
                        * (output_list[_i] - output_list[_j]),
                        -1,
                    )
                    kl_divergences.append(divergence)
            loss_kl_divergence = torch.stack(kl_divergences, 0).mean()

        reward_list = []
        ll_list = []
        td_list = [td.clone() for _ in range(self.num_paths)]
        for i in range(self.num_paths):
            _encoded_inputs = encoded_inputs.clone()
            _attn = attn.clone()
            _V = V.clone()
            _h_old = h_old.clone()
            fixed = self._precompute(_encoded_inputs, path_index=i)

            outputs, actions = [], []
            j = 0
            mask, mask_first = None, None
            while not (self.shrink_size is None and td_list[i]["done"].all()):
                if j > 1 and j % self.eg_step_gap == 0:
                    mask_attn = mask if self.is_vrp else mask ^ mask_first
                    _encoded_inputs, _ = encoder.change(
                        _attn, _V, _h_old, mask_attn
                    )
                    fixed = self._precompute(_encoded_inputs, path_index=i)

                logprobs, mask = self._get_logprobs(fixed, td_list[i], i)
                if j == 0:
                    mask_first = mask

                action = decode_logprobs(
                    logprobs[:, 0, :],
                    mask,
                    decode_type=decoder_kwargs["decode_type"],
                )
                td_list[i].set("action", action)
                td_list[i] = env.step(td_list[i])["next"]
                outputs.append(logprobs[:, 0, :])
                actions.append(action)
                j += 1

            outputs, actions = torch.stack(outputs, 1), torch.stack(actions, 1)
            reward_list.append(env.get_reward(td, actions))
            ll_list.append(get_log_likelihood(outputs, actions, mask=None))

        return (
            torch.stack(reward_list, 1),
            torch.stack(ll_list, 1),
            loss_kl_divergence,
            actions,
        )


class FixedMDAMPolicy(MDAMPolicy):
    def __init__(
        self,
        embed_dim: int = 128,
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        normalization: str = "batch",
        env_name: str = "tsp",
        num_paths: int = 5,
        eg_step_gap: int = 200,
        tanh_clipping: float = 10.0,
        **kwargs,
    ):
        decoder = FixedMDAMDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_paths=num_paths,
            env_name=env_name,
            eg_step_gap=eg_step_gap,
            tanh_clipping=tanh_clipping,
            **kwargs,
        )
        super().__init__(
            decoder=decoder,
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            normalization=normalization,
            env_name=env_name,
        )
        # RL4CO 0.6.0's encoder.change() expects these two aliases, while
        # MultiHeadAttentionMDAM exposes n_heads/embed_dim instead.
        attention_layer = getattr(self.encoder, "attention_layer", None)
        if attention_layer is not None:
            if not hasattr(attention_layer, "num_heads"):
                attention_layer.num_heads = attention_layer.n_heads
            if not hasattr(attention_layer, "val_dim"):
                attention_layer.val_dim = attention_layer.embed_dim
