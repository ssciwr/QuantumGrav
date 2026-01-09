from typing import Any, Sequence
from pathlib import Path
import torch
from ..gnn_model import instantiate_type
from .node_update_GRU import NodeUpdateGRU
from .. import base
#
import jsonschema

class AutoregressiveDecoder(torch.nn.Module, base.Configurable):
    """
    Autoregressive decoder for causal set generation.

    This module reconstructs a causal set link matrix (and optionally node
    features) from a latent vector. It operates sequentially: nodes are generated
    one at a time, and parent relationships are inferred in inverse topological
    order. This mirrors causal set sequential growth algorithms. The decoder consists 
    of several components:

        • node-state creator backbone:
            A single-layer GRU node update block applied at every decoding step to propagate
            information among already generated nodes.

        • parent_logit_mlp:
            A module that computes logits for possible parents of the newly
            generated node. Inputs typically include:
                - provisional state of the new node,
                - states of existing nodes,
                - latent vector z,
                - positional embeddings.

        • decoder_init:
            Optional network mapping latent z → initial hidden state for node 0.

        • ancestor_suppression:
            Mechanism applying multiplicative suppression to parent probabilities
            based on already established ancestor relations. Used to enforce 
            reduction of the link matrix, and therefore transitivity.

        • node_feature_decoder:
            Optional head that decodes final node representations into explicit
            node‑feature vectors.

    The module can operate in training mode using teacher forcing, or in evaluation
    mode by sampling parent sets autoregressively. It outputs:
        - link matrix L,
        - optional decoded node features,
        - optional total log‑likelihood (during teacher‑forcing training).

    This decoder is configured exclusively through json‑schema–validated configs and
    supports modular substitution of all internal components.
    """

    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "DecoderModule Configuration",
        "type": "object",
        "properties": {
            "gru_type": {
                "description": "GRU class for NodeUpdateGRU backbone."
            },
            "gru_args": {
                "type": "array",
                "items": {},
                "description": "Positional arguments for gru_type."
            },
            "gru_kwargs": {
                "type": "object",
                "description": "Keyword arguments for gru_type."
            },
            "gru_aggregation_method": {
                "type": "string",
                "description": "Aggregation method for NodeUpdateGRU"
            },
            "gru_pooling_mlp_type": {
                "description": "Optional MLP used when GRU pooling aggregation='mlp'"
            },
            "gru_pooling_mlp_args": {
                "type": "array",
                "items": {},
                "description": "Arguments for pooling MLP"
            },
            "gru_pooling_mlp_kwargs": {
                "type": "object",
                "description": "Keyword arguments for pooling MLP"
            },

            "parent_logit_mlp_type": {
                "description": "Type of the MLP computing parent logits."
            },
            "parent_logit_mlp_args": {
                "type": "array",
                "items": {},
                "description": "Positional args for parent_logit_mlp_type."
            },
            "parent_logit_mlp_kwargs": {
                "type": "object",
                "description": "Keyword args for parent_logit_mlp_type."
            },

            "decoder_init_type": {
                "description": "Optional initialization network mapping latent → decoder hidden state.",
                "anyOf": [{"type": "null"}, {}]
            },
            "decoder_init_args": {
                "anyOf": [{"type": "null"}, {"type": "array", "items": {}}],
                "description": "Positional args for decoder_init_type."
            },
            "decoder_init_kwargs": {
                "anyOf": [{"type": "null"}, {"type": "object"}],
                "description": "Keyword args for decoder_init_type."
            },

            "node_feature_decoder_type": {
                "description": "Optional head used to decode node features.",
                "anyOf": [{"type": "null"}, {}]
            },
            "node_feature_decoder_args": {
                "anyOf": [{"type": "null"}, {"type": "array", "items": {}}],
                "description": "Positional args for node_feature_decoder_type."
            },
            "node_feature_decoder_kwargs": {
                "anyOf": [{"type": "null"}, {"type": "object"}],
                "description": "Keyword args for node_feature_decoder_type."
            },

            "ancestor_suppression_strength": {
                "type": "number",
                "description": "Coefficient controlling multiplicative suppression of ancestors."
            },
        },
        "required": ["gru_type", "parent_logit_mlp_type"],
        "additionalProperties": False
    }

    def __init__(
        self,
        gru_type: type | torch.nn.Module,
        parent_logit_mlp_type: type | torch.nn.Module,
        gru_args: Sequence[Any] | None = None,
        gru_kwargs: dict[str, Any] | None = None,
        gru_aggregation_method: str = "mean",
        gru_pooling_mlp_type: type | torch.nn.Module | None = None,
        gru_pooling_mlp_args: Sequence[Any] | None = None,
        gru_pooling_mlp_kwargs: dict[str, Any] | None = None,
        parent_logit_mlp_args: Sequence[Any] | None = None,
        parent_logit_mlp_kwargs: dict[str, Any] | None = None,
        decoder_init_type: type | torch.nn.Module | None = None,
        decoder_init_args: Sequence[Any] | None = None,
        decoder_init_kwargs: dict[str, Any] | None = None,
        ancestor_suppression_strength: float = 1.0,
        node_feature_decoder_type: type | torch.nn.Module | None = None,
        node_feature_decoder_args: Sequence[Any] | None = None,
        node_feature_decoder_kwargs: dict[str, Any] | None = None,
    ):
        """
        Args:
            gru_type: Class or module implementing the decoder GNN backbone.
            parent_logit_mlp_type: Class or module producing parent logits.
            gru_args: Positional args for gru_type.
            gru_kwargs: Keyword args for gru_type.
            parent_logit_mlp_args: Positional args for parent_logit_mlp_type.
            parent_logit_mlp_kwargs: Keyword args for parent_logit_mlp_type.
            decoder_init_type: Optional class for initializing decoder state.
            decoder_init_args: Positional args.
            decoder_init_kwargs: Keyword args.
            ancestor_suppression_strength: Transitive-suppression coefficient.
            node_feature_decoder_type: Optional class decoding node features.
            node_feature_decoder_args: Positional args.
            node_feature_decoder_kwargs: Keyword args.
        """
        super().__init__()

        # Store all inputs before building modules
        self.gru_type = gru_type
        self.gru_args = gru_args
        self.gru_kwargs = gru_kwargs
        self.gru_aggregation_method = gru_aggregation_method
        self.gru_pooling_mlp_type = gru_pooling_mlp_type
        self.gru_pooling_mlp_args = gru_pooling_mlp_args
        self.gru_pooling_mlp_kwargs = gru_pooling_mlp_kwargs

        self.parent_logit_mlp_type = parent_logit_mlp_type
        self.parent_logit_mlp_args = parent_logit_mlp_args
        self.parent_logit_mlp_kwargs = parent_logit_mlp_kwargs

        self.decoder_init_type = decoder_init_type
        self.decoder_init_args = decoder_init_args
        self.decoder_init_kwargs = decoder_init_kwargs

        self.ancestor_suppression_strength = ancestor_suppression_strength

        self.node_feature_decoder_type = node_feature_decoder_type
        self.node_feature_decoder_args = node_feature_decoder_args
        self.node_feature_decoder_kwargs = node_feature_decoder_kwargs

        # decoder backbone (NodeUpdateGRU)
        self.node_updater = NodeUpdateGRU(
            gru_type=gru_type,
            gru_args=gru_args,
            gru_kwargs=gru_kwargs or {},
            aggregation_method=gru_aggregation_method,
            pooling_mlp_type=gru_pooling_mlp_type,
            pooling_mlp_args=gru_pooling_mlp_args,
            pooling_mlp_kwargs=gru_pooling_mlp_kwargs,
        )

        # parent logits
        self.parent_logit_mlp = instantiate_type(
            parent_logit_mlp_type,
            parent_logit_mlp_args,
            parent_logit_mlp_kwargs,
        )

        # decoder initial state
        self.decoder_init = (
            instantiate_type(decoder_init_type, decoder_init_args, decoder_init_kwargs)
            if decoder_init_type is not None else None
        )

        # Boolean flag: whether ancestor suppression logic is active
        self.ancestor_suppression = (ancestor_suppression_strength != 0)

        # Warn if value is outside the interpretable range [0, 1]
        if self.ancestor_suppression_strength < 0 or self.ancestor_suppression_strength > 1:
            print(
                f"Warning: ancestor_suppression_strength={self.ancestor_suppression_strength} "
                "is outside the typical range [0, 1]. Values >1 or <0 are not theoretically well-understood "
                "and may lead to unstable or unintuitive suppression behaviour."
            )

        # Warn if suppression is fully disabled
        if not self.ancestor_suppression:
            print(
                "Warning: ancestor_suppression_strength=0 → "
                "ancestor-suppression pipeline is disabled (no transitivity-based suppression will occur)."
            )

        # node feature decoder
        self.node_feature_decoder = (
            instantiate_type(
                node_feature_decoder_type,
                node_feature_decoder_args,
                node_feature_decoder_kwargs,
            ) if node_feature_decoder_type is not None else None
        )


    def init_state(self, z: torch.Tensor) -> torch.Tensor:
        """Initialize decoder hidden state from latent z."""
        if self.decoder_init is None:
            return z
        return self.decoder_init(z)

    def forward(
        self,
        z: torch.Tensor,
        teacher_forcing_targets: torch.Tensor | None = None,
        atom_count: int | None = None,
    ):
        """
        Batch‑aware wrapper around autoregressive_decode().
        Supports:
            - z.dim() == 1 → decode single causal set
            - z.dim() == 2 → sequentially decode each latent vector
        """
        # Single latent vector
        if z.dim() == 1:
            return self.autoregressive_decode(
                z, teacher_forcing_targets=teacher_forcing_targets, atom_count=atom_count
            )

        # Batched latents: process sequentially
        if z.dim() == 2:
            batches = z.size(0)
            outputs = []
            for batch in range(batches):
                z_b = z[batch].view(-1)   # ensure 1‑D latent for autoregressive_decode

                if teacher_forcing_targets is not None:
                    tgt_b = teacher_forcing_targets[batch]
                else:
                    tgt_b = None

                outputs.append(
                    self.autoregressive_decode(
                        z_b,
                        teacher_forcing_targets=tgt_b,
                        atom_count=atom_count,
                    )
                )
            return outputs

        raise ValueError(
            f"Decoder expected z to have dim 1 or 2, but got shape {tuple(z.shape)}."
        )

    def autoregressive_decode(
        self,
        z: torch.Tensor,
        teacher_forcing_targets: torch.Tensor | None = None,
        atom_count: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Autoregressive CST-style decoder:
        - grows nodes sequentially
        - uses bitset ancestor tracking
        - processes candidate parents in inverse topological order
        - applies stochastic sampling with ancestor suppression
        - returns (adjacency, node_states)
        """

        # Enforce teacher forcing during training
        if self.training and teacher_forcing_targets is None:
            raise ValueError(
                "Decoder is in training mode but teacher_forcing_targets=None. "
                "Training the autoregressive decoder requires teacher-forcing targets. "
                "Use eval() mode for sampling/generation."
            )

        # latent must be a 1D tensor [latent_dim]
        if z.dim() != 1:
            raise ValueError(
                f"Expected latent vector z to be 1-D (shape [latent_dim]), but got shape {tuple(z.shape)}. "
                "Please pass z.squeeze(0) if you produced a batch of size 1."
            )
                
        # Determine N_max (number of atoms to generate).
        # Teacher forcing always determines N_max. If atom_count is also provided, warn and ignore it.
        if teacher_forcing_targets is not None:
            if atom_count is not None:
                print(
                    "Warning: Both atom_count and teacher_forcing_targets were provided. "
                    "teacher_forcing_targets determines N_max; atom_count is ignored."
                )
            N_max = teacher_forcing_targets.size(0)
        else:
            # No teacher forcing → atom_count must be provided.
            if atom_count is None:
                raise ValueError(
                    "Decoder requires atom_count when teacher_forcing_targets is not provided."
                )
            N_max = atom_count

        # initial structures
        h0 = self.init_state(z)
        hprev = h0.clone()
        node_states = []                      # list of tensors
        links = torch.zeros((N_max, N_max), dtype=torch.float32, device=h0.device)  # preallocated adjacency tensor
        step_logprobs = []                      # accumulate per-step log likelihoods during training
        parent_indices = torch.empty(0, dtype=torch.long, device=h0.device)
        if self.ancestor_suppression:
            # Preallocate ancestor matrix A: A[t, j]=True iff j is ancestor of t
            A = torch.zeros((N_max, N_max), dtype=torch.bool, device=h0.device)

        # grow nodes 1 .. N_max-1
        for t in range(1, N_max):
            prev_count = t

            # create new node state
            if len(parent_indices) == 0:
                parent_states = torch.empty((0, self.node_updater.hidden_dim), device=h0.device)
                new_state = h0.clone()
            else:
                parent_states = h_prev[parent_indices]
                new_state = self.node_updater(parent_states)      # GRU creates node state from parent nodes
            node_states.append(new_state)

            # Prepare inputs for parent_logit_mlp
            h_prev = torch.stack(node_states, dim=0)
            z_rep = z.unsqueeze(0).repeat(prev_count, 1)
            parent_mlp_in = torch.cat([h_prev, z_rep], dim=1)

            # compute parent logits for all previous nodes
            logits = self.parent_logit_mlp(parent_mlp_in).squeeze(-1)
            probs = torch.sigmoid(logits)

            if self.ancestor_suppression:
                # Use ancestor matrix A for suppression
                A_slice = A[:prev_count, :prev_count]
                D = A_slice.t()
                P = probs.unsqueeze(0).expand_as(D)
                suppression = torch.prod(torch.where(D, 1 - P, torch.ones_like(P)), dim=1)
                adjusted_probs = probs * suppression
            else:
                adjusted_probs = probs

            # teacher forcing vs. sampling
            if teacher_forcing_targets is not None:
                gt_row = teacher_forcing_targets[t][:prev_count].to(h0.device)
                parent_mask = gt_row.to(torch.bool)

                # accumulate log likelihood for this step
                eps = 1e-12
                log_p = (
                    gt_row * torch.log(adjusted_probs + eps)
                    + (1 - gt_row) * torch.log(1 - adjusted_probs + eps)
                )
                step_logprobs.append(log_p.sum())
            else:
                parent_mask = torch.bernoulli(adjusted_probs).to(torch.bool)
            
            parent_indices = (parent_mask == 1).nonzero(as_tuple=False).flatten()

            # update adjacency structure in preallocated tensor
            links[t, :prev_count] = parent_mask.to(torch.float32)

            # update ancestor matrix A: row t
            # A[t] = OR over rows of parents; then set direct parents
            if self.ancestor_suppression:
                if len(parent_indices) > 0:
                    # Compute new ancestor row from parents
                    new_row = torch.any(A[parent_indices], dim=0)

                    # Mark direct parents as ancestors
                    direct_parents_mask = torch.zeros_like(new_row)
                    direct_parents_mask[parent_indices] = True
                    new_row = new_row | direct_parents_mask

                    # Replace row t without touching the original tensor in-place:
                    # A = A.clone() ensures autograd sees a fresh version counter.
                    A = A.clone()
                    A[t] = new_row

        if self.node_feature_decoder is not None:
            # Final GRU update: ensure node_states reflect the full graph including last-node parents
            if len(parent_indices) == 0:
                parent_states = torch.empty((0, self.node_updater.hidden_dim), device=h0.device)
                new_state = h0.clone()
            else:
                parent_states = h_prev[parent_indices]
                new_state = self.node_updater(parent_states)      # GRU creates node state from parent nodes
            node_states.append(new_state)

        X_hidden = torch.stack(node_states, dim=0)
        X_out = self.node_feature_decoder(X_hidden) if self.node_feature_decoder is not None else None
        # Determine whether to compute log-likelihood
        if teacher_forcing_targets is not None and len(step_logprobs) > 0:
            total_logprob = torch.stack(step_logprobs).sum()
        else:
            total_logprob = None

        return links, X_out, total_logprob

    def reconstruct_link_matrix(self, z, atom_count):
        out = self.forward(z, atom_count=atom_count)
        if z.dim() == 1:
            L, _, _ = out
            return L
        else:
            return [L for (L, _, _) in out]

    def reconstruct_node_features(self, z, atom_count):
        if self.node_feature_decoder is None:
            raise RuntimeError("Decoder not configured with node_feature_decoder.")

        out = self.forward(z, atom_count=atom_count)

        if z.dim() == 1:
            L, X, _ = out
            return L, X

        else:
            return [(L, X) for (L, X, _) in out]

    def compute_normalized_log_likelihood(self, z, teacher_forcing_targets):
        out = self.forward(z, teacher_forcing_targets=teacher_forcing_targets)

        if z.dim() == 1:
            _, _, logprob = out
            n = teacher_forcing_targets.size(0)
            return logprob / (n * (n - 1) / 2)

        else:
            results = []
            for (L, X, logp), tgt in zip(out, teacher_forcing_targets):
                n = tgt.size(0)
                results.append(logp / (n * (n - 1) / 2))
            return torch.stack(results, dim=0)
    
    @classmethod
    def from_config(cls, config: dict) -> "AutoregressiveDecoder":
        """
        Instantiate an AutoregressiveDecoder from a configuration dictionary.

        The config must satisfy the class schema. This method validates the
        config using jsonschema before constructing the decoder.
        """
        jsonschema.validate(instance=config, schema=cls.schema)

        pl_type  = config["parent_logit_mlp_type"]
        pl_args  = config.get("parent_logit_mlp_args", None)
        pl_kwargs = config.get("parent_logit_mlp_kwargs", None)

        init_type  = config.get("decoder_init_type", None)
        init_args  = config.get("decoder_init_args", None)
        init_kwargs = config.get("decoder_init_kwargs", None)

        nf_type  = config.get("node_feature_decoder_type", None)
        nf_args  = config.get("node_feature_decoder_args", None)
        nf_kwargs = config.get("node_feature_decoder_kwargs", None)

        gru_agg = config.get("gru_aggregation_method", "mean")
        gru_p_mlp_type = config.get("gru_pooling_mlp_type", None)
        gru_p_mlp_args = config.get("gru_pooling_mlp_args", None)
        gru_p_mlp_kwargs = config.get("gru_pooling_mlp_kwargs", None)

        return cls(
            gru_type=config["gru_type"],
            parent_logit_mlp_type=pl_type,

            gru_args=config.get("gru_args", None),
            gru_kwargs=config.get("gru_kwargs", None),
            gru_aggregation_method=gru_agg,
            gru_pooling_mlp_type=gru_p_mlp_type,
            gru_pooling_mlp_args=gru_p_mlp_args,
            gru_pooling_mlp_kwargs=gru_p_mlp_kwargs,

            parent_logit_mlp_args=pl_args,
            parent_logit_mlp_kwargs=pl_kwargs,

            decoder_init_type=init_type,
            decoder_init_args=init_args,
            decoder_init_kwargs=init_kwargs,

            ancestor_suppression_strength=config.get("ancestor_suppression_strength", 1.0),

            node_feature_decoder_type=nf_type,
            node_feature_decoder_args=nf_args,
            node_feature_decoder_kwargs=nf_kwargs,
        )
    
    def save(self, path: str | Path) -> None:
        """Save only decoder weights."""
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path, config: dict, device=torch.device("cpu")) -> "AutoregressiveDecoder":
        """Load decoder from state_dict + external config."""
        model = cls.from_config(config).to(device)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        return model