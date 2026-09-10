"""MDAM variant with the path-diversity KL term from the original paper."""

from __future__ import annotations

from rl4co.models import MDAM


class MDAMWithKL(MDAM):
    """RL4CO MDAM plus ``- kl_loss * KL`` regularization.

    RL4CO's MDAM computes the first-step path KL internally and exposes it as
    ``policy_out["entropy"]``, but its default training loss ignores it.  The
    original MDAM training script backpropagates this term before the
    REINFORCE loss, which is equivalent to adding it to the total loss.
    """

    def __init__(self, *args, kl_loss: float = 0.0, **kwargs):
        self.kl_loss = float(kl_loss)
        super().__init__(*args, **kwargs)

    def calculate_loss(self, td, batch, policy_out, reward=None, log_likelihood=None):
        out = super().calculate_loss(
            td,
            batch,
            policy_out,
            reward=reward,
            log_likelihood=log_likelihood,
        )
        kl = out.get("entropy")
        if kl is not None and self.kl_loss != 0.0:
            kl_term = self.kl_loss * kl
            out["kl_loss"] = kl_term
            out["loss"] = out["loss"] - kl_term
        return out
