import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ------------------------------------------------------------------
# Replace the old Gate with a differentiable straight-through gate
# ------------------------------------------------------------------
class DiffGate(nn.Module):
    """
    Binary gate trained with the relaxed-Bernoulli (Concrete) trick.
       τ - temperature; anneal from 1.0 → 0.1 during training if you like
       hard=True  →  straight-through (skip HeavyNet when g=0)
    """

    def __init__(self, temperature: float = 1.0, hard: bool = True):
        super().__init__()
        self.temperature = temperature
        self.hard = hard
        self.net = nn.Sequential(
            layer_init(nn.Conv2d(3, 4, 1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(7 * 7 * 4, 2), std=1),
        )

    def forward(self, x, action=False):
        logits = self.net(x.permute(0, 3, 1, 2)).squeeze(-1)

        categorization = nn.functional.gumbel_softmax(logits)
        return categorization[:, -1]


################################################################################
# NEW adaptive-compute architecture
################################################################################
class CheapNet(nn.Module):
    "4-number arrow-aware MLP (same as in the previous answer, 10 k params)"

    def __init__(self, obs_space, act_n):
        super().__init__()
        self.actor = layer_init(nn.Linear(1, act_n), std=0.01)
        self.critic = layer_init(nn.Linear(1, 1), std=1.0)

    def forward(self, x, arrow):
        x = x.view(x.size(0), -1)
        logits = self.actor(arrow)
        value = self.critic(arrow)
        return logits, value


class HeavyNet(nn.Module):
    def __init__(self, act_n):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, 3, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(288, 64)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(64, act_n), std=0.01)
        self.critic = layer_init(nn.Linear(64, 1), std=1)

    def forward(self, x):
        # x: (B,7,7,3)
        x = x.permute(0, 3, 1, 2)  # to NCHW
        feat = self.network(x)
        logits = self.actor(feat)
        value = self.critic(feat)
        return logits, value


# class HeavyNet(nn.Module):
#     "CNN + GRU power-house that can handle the POMDP when no arrow exists"

#     def __init__(self, act_n):
#         super().__init__()
#         self.cnn = nn.Sequential(
#             layer_init(nn.Conv2d(3, 32, 3)),
#             nn.ReLU(),
#             layer_init(nn.Conv2d(32, 32, 3)),
#             nn.ReLU(),
#             nn.Flatten(),
#             layer_init(nn.Linear(288, 64)),
#             nn.ReLU(),
#         )
#         self.rnn = nn.GRU(64, 128, batch_first=True)
#         self.actor = layer_init(nn.Linear(128, act_n), 0.01)
#         self.critic = layer_init(nn.Linear(128, 1), 1)
#         self._h0 = nn.Parameter(torch.zeros(1, 1, 128))  # learnt initial hidden

#     def forward(self, x):
#         # x: (B,7,7,3)
#         x = x.permute(0, 3, 1, 2)  # to NCHW
#         feat = self.cnn(x)
#         out, _ = self.rnn(feat.unsqueeze(1), self._h0.repeat(1, x.size(0), 1))
#         hid = out.squeeze(1)
#         logits = self.actor(hid)
#         value = self.critic(hid)
#         return logits, value


class GatedAgent(nn.Module):
    def __init__(self, envs, compute_penalty=0.003):
        super().__init__()
        self.cheap = CheapNet(envs.observation_space, envs.action_space.n)
        self.heavy = HeavyNet(envs.action_space.n)
        self.gate = DiffGate()

    # --------------- helper: split forward through the two experts ------------
    def _run_branch(self, cheap_mask, obs, arrow):
        """
        cheap_mask ∈ {0,1}^B   1 → run cheap ;  0 → run heavy
        Only the selected branch is actually executed ⇒ ∆ compute!
        Returns: (logits,value)
        """
        if cheap_mask.all():
            return self.cheap(obs, arrow)
        if (~cheap_mask).all():
            return self.heavy(obs)

        # Mixed batch - do them separately and stitch together
        logits = torch.zeros(
            (obs.size(0), self.cheap.actor.out_features), device=obs.device
        )
        values = torch.zeros(obs.size(0), 1, device=obs.device)

        idx_c = cheap_mask.nonzero(as_tuple=True)[0]
        idx_h = (~cheap_mask).nonzero(as_tuple=True)[0]
        logits[idx_c], values[idx_c] = self.cheap(obs[idx_c])
        logits[idx_h], values[idx_h] = self.heavy(obs[idx_h])
        return logits, values

    # --------------- public API used by the PPO loop --------------------------
    def get_value(self, x, arrow):
        # Call gate once, create a hard mask, and run only the selected branch.
        with torch.no_grad():
            p_cheap = self.gate(x, arrow)  # probability of CHEAP branch
        cheap_mask = (p_cheap > 0.5).bool()  # hard decision (no grads anyway)
        return self._run_branch(cheap_mask, x, arrow)[1]

    def get_action_and_value(self, x, arrow=None, action=None):
        # 1) Sample gate and turn it into a hard mask

        p_cheap = self.gate(x, arrow, action=True)  # (B,) - probability CHEAP
        cheap_mask = (p_cheap > 0.5).bool()  # (B,)  - True ⇒ CheapNet

        # 2) Run *only* the branch required for every element of the batch
        logits, value = self._run_branch(cheap_mask, x, arrow)

        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()

        # add gate log-prob so PPO treats the gate as part of the policy
        logp_gate = torch.where(
            cheap_mask, torch.log(p_cheap + 1e-8), torch.log(1.0 - p_cheap + 1e-8)
        )
        logp = dist.log_prob(action) + logp_gate

        # entropy = entropy(action) + entropy(gate)
        entropy_gate = -(
            p_cheap * torch.log(p_cheap + 1e-8)
            + (1 - p_cheap) * torch.log(1 - p_cheap + 1e-8)
        )
        entropy = dist.entropy() + entropy_gate

        # For bookkeeping downstream (e.g. compute-penalty)
        self._last_gate = cheap_mask.float()

        return action, logp, entropy, value
