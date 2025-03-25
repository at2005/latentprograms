import torch
from tqdm import tqdm


# inference time generation
MAX_SEQ_LEN = 1000
NUM_ITERS = 100
NUM_STEPS = 200
NUM_GRPO_ITERS = 10
num_groups = 128
group_size = 64
num_tokens = 300
epsilon = 0.2
beta = 0.3
lr = 3e-4

def grader():
    pass

def get_advantages(tokens) -> torch.Tensor:
    with torch.no_grad():
        rewards: torch.Tensor = grader(
            tokens
        )  # rewards of dimension (B, G), ie one for each term in the group for each batch
        mean_reward = torch.mean(rewards, dim=-1)  # across each group
        reward_std = torch.std(rewards, dim=-1)  # across each group
        advantages = (rewards - mean_reward)  # get (B, G)
        advantages = advantages.unsqueeze(-1)  # (B, G, 1)
        return advantages


def grpo_loss(
    advantages, new_priors, old_priors, reference_priors, token_ids, filter_fn=None
):
    # new priors are of shape (B, G, T, V)
    ratio = new_priors[:, :, token_ids, :] / old_priors[:, :, token_ids, :]
    min_clipped_advantage = torch.min(
        advantages * ratio,  # B, G, 1 *  B, G, T checks out
        torch.clip(ratio, 1 - epsilon, 1 + epsilon) * advantages,
    )

    # filter function for computing kl_divergence loss so that the only drift from the OG model
    # is constrained to actual tokens (ie the original pretrained policy)
    if filter_fn:
        mask = filter_fn(token_ids)
        token_ids = token_ids[mask]

    ref_ratio = reference_priors[:, :, token_ids, :] / new_priors[:, :, token_ids, :]
    kl_loss = ref_ratio - torch.log(ref_ratio) - 1.0  # B, G, T, 1
    objective: torch.Tensor = min_clipped_advantage - beta * kl_loss

    # mean across T, across G, and B
    loss = -objective.sum(dim=2).mean(dim=[0, 1])
    return loss