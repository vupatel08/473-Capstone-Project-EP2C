## utils.py

import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List, Tuple

# Import configuration parameters from the provided config.yaml
# Here, for demonstration, we're assuming they are passed or accessible as constants.
# In actual usage, you may import from a config module or read from the YAML.
# For this code snippet, we'll define defaults; ensure to override with your config.
DEFAULT_BETA = 0.1
DEFAULT_ALPHA = 0.5
EPSILON = 1e-8

def generate_response(
    model,
    prompt: str,
    max_tokens: int,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 50
) -> str:
    """
    Generates a response from the model conditioned on the prompt.
    """
    generation_kwargs = {
        "do_sample": True,
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "eos_token_id": model.tokenizer.eos_token_id,
        "pad_token_id": model.tokenizer.eos_token_id
    }
    input_ids = model.tokenizer.encode(prompt, return_tensors='pt').to(next(model.model.parameters()).device)
    with torch.no_grad():
        output_ids = model.model.generate(input_ids, **generation_kwargs)
    response = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

def get_response_probabilities(
    model,
    context_tokens: torch.LongTensor,
    target_tokens: torch.LongTensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the probability distribution over the vocabulary
    for each token in target_tokens conditioned on context_tokens.
    Returns two tensors: model_probs and ref_probs, each [T, vocab_size].
    """
    T = target_tokens.shape[0]
    device = next(model.model.parameters()).device
    model_probs_list = []
    ref_probs_list = []

    for t in range(T):
        # Build input: context + tokens[:t]
        context_ids = context_tokens.tolist()
        prefix_ids = context_ids + target_tokens[:t].tolist()
        input_ids = torch.tensor([prefix_ids], device=device)
        # Model forward to get logits
        with torch.no_grad():
            outputs = model.model(**{"input_ids": input_ids})
            logits = outputs.logits  # shape [1, seq_len, vocab]
        last_logits = logits[0, -1, :]  # last token logits
        probs = F.softmax(last_logits, dim=-1)  # shape [vocab_size]
        model_probs_list.append(probs)

        # For ref_probs, same process
        with torch.no_grad():
            ref_outputs = model.ref_model.model(**{"input_ids": input_ids})
            ref_logits = ref_outputs.logits
        ref_probs = F.softmax(ref_logits[0, -1, :], dim=-1)
        ref_probs_list.append(ref_probs)

    model_probs = torch.stack(model_probs_list, dim=0)  # [T, vocab_size]
    ref_probs = torch.stack(ref_probs_list, dim=0)
    return model_probs, ref_probs

def kl_divergence(p_probs: torch.Tensor, q_probs: torch.Tensor) -> torch.Tensor:
    """
    Computes the KL divergence D_KL(p || q) for two probability distributions p and q.
    p_probs and q_probs shape: [vocab_size]
    """
    p_probs = p_probs + EPSILON
    q_probs = q_probs + EPSILON
    kl = torch.sum(p_probs * (torch.log(p_probs) - torch.log(q_probs)))
    return kl

def sequence_kl_divergence(
    p_probs_seq: torch.Tensor,
    q_probs_seq: torch.Tensor
) -> torch.Tensor:
    """
    Computes the sequence KL divergence as sum over token-level KLs.
    p_probs_seq, q_probs_seq shape: [T, vocab_size]
    """
    kl_sum = 0.0
    T = p_probs_seq.shape[0]
    for t in range(T):
        kl_sum += kl_divergence(p_probs_seq[t], q_probs_seq[t])
    return kl_sum

def compute_advantage(
    Q_values: torch.Tensor,
    V_value: torch.Tensor
) -> torch.Tensor:
    """
    Computes advantage at each token: A(s,a) = Q(s,a) - V(s).
    Inputs:
        Q_values: Tensor [T], estimated Q at each token.
        V_value: scalar estimate of V for the state.
    Output:
        advantages: Tensor [T]
    """
    advantages = Q_values - V_value
    return advantages

def preference_probability(u: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """
    Computes preference probability P_{BT} = sigmoid(u - delta).
    Args:
        u: scalar or tensor, reward difference.
        delta: scalar or tensor, divergence difference.
    """
    return torch.sigmoid(u - delta)

def compute_token_reward(response_text: str, response_score: float = 1.0) -> float:
    """
    Placeholder for token-level reward based on human evaluation, GPT scoring, etc.
    Here, for simulation, return a scaled score.
    """
    return response_score  # or any function mapping responses to scalar reward.

def estimate_Q(
    tokenized_response: torch.LongTensor,
    token_rewards: List[float]
) -> torch.Tensor:
    """
    Estimate Q-values at each token by summing subsequent token rewards (or proxy).
    For simplicity, assume Q at token t is sum of rewards from t to end.
    """
    T = len(token_rewards)
    Q_t = torch.zeros(T)
    cumulative = 0.0
    for t in reversed(range(T)):
        cumulative += token_rewards[t]
        Q_t[t] = cumulative
    return Q_t

def estimate_V(Q_values: torch.Tensor, mask: torch.Tensor) -> float:
    """
    Estimate V as average Q over tokens, masked.
    """
    total = torch.sum(Q_values * mask)
    count = mask.sum() + EPSILON
    return total / count

def plot_frontier(rewards: List[float], kl_vals: List[float], title: str):
    """
    Plot reward vs KL divergence frontier.
    """
    plt.figure()
    plt.plot(kl_vals, rewards, marker='o')
    plt.xlabel('KL Divergence')
    plt.ylabel('Reward')
    plt.title(title)
    plt.grid()
    plt.show()

def plot_divergence_curves(
    preferred_div: List[float],
    dispreferred_div: List[float],
    title: str
):
    """
    Plot the divergence trends over training steps.
    """
    plt.figure()
    steps = list(range(len(preferred_div)))
    plt.plot(steps, preferred_div, label='Preferred')
    plt.plot(steps, dispreferred_div, label='Dispreferred')
    plt.xlabel('Training Step')
    plt.ylabel('Seq KL Divergence')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def compute_stop_gradient(value: torch.Tensor) -> torch.Tensor:
    """
    Returns the tensor with gradient stopped.
    """
    return value.detach()

def normalize_probs(probs: torch.Tensor) -> torch.Tensor:
    """
    Normalize probabilities across the vocabulary.
    """
    sum_probs = probs.sum()
    return probs / (sum_probs + EPSILON)
