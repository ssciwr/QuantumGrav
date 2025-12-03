from typing import Any, Callable, Sequence
from pathlib import Path
from inspect import isclass
import torch
from . import utils
from . import linear_sequential as QGLS
from . import gnn_block as QGGNN

class ModuleWrapper(torch.nn.Module):
    """Wrapper to make pooling functions compatible with ModuleList."""

    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    def get_fn(self) -> Callable:
        return self.fn

class EncoderModule(torch.nn.Module):
    """
    Encoder module:
      - sequence of GNNBlocks
      - optional pooling layers
      - optional aggregate pooling
      - optional graph_features_net
      - optional aggregate_graph_features
    """

    def __init__(
        self,
        encoder: Sequence[QGGNN.GNNBlock],
        pooling_layers: Sequence[torch.nn.Module] | None = None,
        aggregate_pooling: torch.nn.Module | Callable | None = None,
        graph_features_net: torch.nn.Module | None = None,
        aggregate_graph_features: torch.nn.Module | Callable | None = None,
    ):
        super().__init__()

        # encoder is a sequence of GNN blocks. There must be at least one
        self.encoder = torch.nn.ModuleList(encoder)

        if len(self.encoder) == 0:
            raise ValueError("At least one GNN block must be provided.")

        # set up pooling layers and their aggregation
        if pooling_layers is not None:
            if len(pooling_layers) == 0:
                raise ValueError("EncoderModule: At least one pooling layer must be provided..")
            
            self.pooling_layers = torch.nn.ModuleList(
                [
                    p
                    if isclass(type(p)) and issubclass(type(p), torch.nn.Module) 
                    else ModuleWrapper(p)
                    for p in pooling_layers
                ]
            )
        else:
            self.pooling_layers = None

        # aggregate pooling layer
        self.aggregate_pooling = aggregate_pooling
        
        if aggregate_pooling is not None:
            if not isclass(aggregate_pooling) or not issubclass(
                aggregate_pooling, torch.nn.Module
            ):
                self.aggregate_pooling = ModuleWrapper(aggregate_pooling)

        pooling_funcs = [self.aggregate_pooling, self.pooling_layers]
        if any([p is not None for p in pooling_funcs]) and not all(
            p is not None for p in pooling_funcs
        ):
            raise ValueError(
                "If pooling layers are to be used, both an aggregate pooling method and pooling layers must be provided."
            )
        
        # set up graph features processing if provided
        self.graph_features_net = graph_features_net

        self.aggregate_graph_features = aggregate_graph_features

        if aggregate_graph_features is not None:
            if not isclass(aggregate_graph_features) or not issubclass(
                aggregate_graph_features, torch.nn.Module
            ):
                self.aggregate_graph_features = ModuleWrapper(aggregate_graph_features)

        graph_processors = [self.graph_features_net, self.aggregate_graph_features]

        if any([g is not None for g in graph_processors]) and not all(
            g is not None for g in graph_processors
        ):
            raise ValueError(
                "If graph features are to be used, both a graph features network and an aggregation method must be provided."
            )

        # encoder output dimension
        # if aggregate_graph_features is nn.Module, use its final layer
        if isinstance(self.aggregate_graph_features, torch.nn.Module) and not isinstance(self.aggregate_graph_features, ModuleWrapper):
            last = self.aggregate_graph_features
            if hasattr(last, "layers") and hasattr(last.layers[-1], "out_features"):
                self.out_dim = last.layers[-1].out_features
            else:
                raise ValueError("Cannot infer encoder output dim from aggregate_graph_features.")
        # elif aggregate_pooling is nn.Module
        elif isinstance(self.aggregate_pooling, torch.nn.Module) and not isinstance(self.aggregate_pooling, ModuleWrapper):
            last = self.aggregate_pooling
            if hasattr(last, "layers") and hasattr(last.layers[-1], "out_features"):
                self.out_dim = last.layers[-1].out_features
            else:
                raise ValueError("Cannot infer encoder output dim from aggregate_pooling.")
        # else fallback to last GNN block output dim
        else:
            last_gnn = self.encoder[-1]
            if hasattr(last_gnn, "out_features"):
                self.out_dim = last_gnn.out_features
            else:
                raise ValueError("Encoder last GNN block has no out_features.")

    def eval_encoder(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        gcn_kwargs: dict[Any, Any] | None = None,
    ) -> torch.Tensor:
        """Evaluate the GCN network on the input data.

        Args:
            x (torch.Tensor): Input node features.
            edge_index (torch.Tensor): Graph connectivity information.
            gcn_kwargs (dict[Any, Any], optional): Additional arguments for the GCN. Defaults to None.

        Returns:
            torch.Tensor: Output of the GCN network.
        """
        # Apply each GCN layer to the input features
        features = x
        for gnn_layer in self.encoder:
            features = gnn_layer(
                features, edge_index, **(gcn_kwargs if gcn_kwargs else {})
            )
        return features

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        gcn_kwargs: dict | None = None,
    ) -> torch.Tensor:
        """Get the **structural graph embedding** produced by the GNN encoder.

        This returns the pooled embedding **before** any graph-feature fusion.

        Args:
            x (torch.Tensor): Input node features.
            edge_index (torch.Tensor): Graph connectivity.
            batch (torch.Tensor | None): Batch vector for pooling.
            gcn_kwargs (dict | None): Additional GNN kwargs.

        Returns:
            torch.Tensor: Structural graph embedding of shape [B, D].
        """
        # apply the GCN backbone to the node features
        embeddings = self.eval_encoder(
            x, edge_index, **(gcn_kwargs if gcn_kwargs else {})
        )

        # pool everything together into a single graph representation
        if self.pooling_layers is not None and self.aggregate_pooling is not None:
            pooled_embeddings = [
                pooling_op(embeddings, batch) for pooling_op in self.pooling_layers
            ]

            return self.aggregate_pooling(pooled_embeddings)
        else:
            return embeddings

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        graph_features: torch.Tensor | None = None,
        embedding_kwargs: dict[Any, Any] | None = None,
    ) -> torch.Tensor:
        """Forward pass of the encoder.

        Runs the structural encoder (GNN + pooling) and then optionally
        processes and fuses graph-level features. The output is the final
        graph embedding that will be fed into the latent module.

        Args:
            x (torch.Tensor): Input node features.
            edge_index (torch.Tensor): Graph connectivity.
            batch (torch.Tensor): Batch vector for pooling.
            graph_features (torch.Tensor | None): Optional graph-level features.
            embedding_kwargs (dict[Any, Any] | None): Extra args for the GNN.

        Returns:
            torch.Tensor: Final graph embedding (structural pooled embedding
            fused with optional graph features).
        """
        # apply the GCN backbone to the node features

        embeddings = self.get_embeddings(
            x, edge_index, batch, gcn_kwargs=embedding_kwargs
        )

        # If we have graph features, we need to process them and concatenate them with the node features
        if graph_features is not None and self.graph_features_net is not None:
            graph_features = self.graph_features_net(graph_features)

        if self.aggregate_graph_features is not None and graph_features is not None:
            embeddings = self.aggregate_graph_features(embeddings, graph_features)

        # downstream tasks are given out as is, no softmax or other assumptions
        return embeddings
    
    @classmethod
    def from_config(cls, config: dict) -> "GNNModel":
        """Create an encoder (GNNModel) from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary containing parameters for the model.

        Returns:
            GNNModel: An instance of GNNModel.
        """

        # create encoder
        encoder = [QGGNN.GNNBlock.from_config(cfg) for cfg in config["encoder"]]

        # make pooling layers
        pooling_layers_cfg = config.get("pooling_layers", None)

        if pooling_layers_cfg is not None:
            pooling_layers = []
            for pool_cfg in pooling_layers_cfg:
                pooling_layer = cls._cfg_helper(
                    pool_cfg,
                    utils.get_registered_pooling_layer,
                    f"The config for a pooling layer is invalid: {pool_cfg}",
                )
                pooling_layers.append(pooling_layer)
        else:
            pooling_layers = None

        # graph aggregation pooling
        aggregate_pooling_cfg = config.get("aggregate_pooling", None)
        if aggregate_pooling_cfg is not None:
            aggregate_pooling = cls._cfg_helper(
                config["aggregate_pooling"],
                utils.get_pooling_aggregation,
                f"The config for 'aggregate_pooling' is invalid: {config['aggregate_pooling']}",
            )
        else:
            aggregate_pooling = None

        # make graph features network and aggregations
        if "graph_features_net" in config and config["graph_features_net"] is not None:
            graph_features_net = QGLS.LinearSequential.from_config(
                config["graph_features_net"]
            )
        else:
            graph_features_net = None

        if graph_features_net is not None:
            aggregate_graph_features = cls._cfg_helper(
                config["aggregate_graph_features"],
                utils.get_graph_features_aggregation,
                f"The config for 'aggregate_graph_features' is invalid: {config['aggregate_graph_features']}",
            )
        else:
            aggregate_graph_features = None

        
        # return the model
        return cls(
            encoder=encoder,
            pooling_layers=pooling_layers,
            graph_features_net=graph_features_net,
            aggregate_graph_features=aggregate_graph_features,
            aggregate_pooling=aggregate_pooling,
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize the model to a config

        Returns:
            dict[str, Any]: _description_
        """
        pooling_layer_names = None
        if self.pooling_layers is not None:
            pooling_layer_names = []
            for layer in self.pooling_layers:
                if isinstance(layer, ModuleWrapper):
                    pooling_layer_names.append(
                        {
                            "type": utils.pooling_layers_names[layer.get_fn()],
                            "args": [],
                            "kwargs": {},
                        }
                    )
                else:
                    pooling_layer_names.append(
                        {
                            "type": utils.pooling_layers_names[layer],
                            "args": [],
                            "kwargs": {},
                        }
                    )

        aggregate_graph_features_names = None
        if self.aggregate_graph_features is not None:
            if isinstance(self.aggregate_graph_features, ModuleWrapper):
                aggregate_graph_features_names = {
                    "type": utils.graph_features_aggregations_names[
                        self.aggregate_graph_features.get_fn()
                    ],
                    "args": [],
                    "kwargs": {},
                }
            else:
                aggregate_graph_features_names = {
                    "type": utils.graph_features_aggregations_names[
                        self.aggregate_graph_features
                    ],
                    "args": [],
                    "kwargs": {},
                }

        aggregate_pooling_names = None
        if self.aggregate_pooling is not None:
            if isinstance(self.aggregate_pooling, ModuleWrapper):
                aggregate_pooling_names = {
                    "type": utils.pooling_aggregations_names[
                        self.aggregate_pooling.get_fn()
                    ],
                    "args": [],
                    "kwargs": {},
                }
            else:
                aggregate_pooling_names = {
                    "type": utils.pooling_aggregations_names[self.aggregate_pooling],
                    "args": [],
                    "kwargs": {},
                }

        
        config = {
            "encoder": [encoder_layer.to_config() for encoder_layer in self.encoder],
            "pooling_layers": pooling_layer_names,
            "graph_features_net": self.graph_features_net.to_config()
            if self.graph_features_net
            else None,
            "aggregate_graph_features": aggregate_graph_features_names,
            "aggregate_pooling": aggregate_pooling_names,
        }

        return config

class DecoderModule(torch.nn.Module):
    """
    Bundles all decoder-related components:
      - GNNBlocks
      - parent_logit_mlp
      - decoder_init + decoder_init_type
      - ancestor_suppression_strength
      - positional embeddings (max_positions)
      - node_feature_decoder + output_node_features
    """

    def __init__(
        self,
        decoder: Sequence[QGGNN.GNNBlock],
        parent_logit_mlp: torch.nn.Module,
        decoder_init: torch.nn.Module | None = None,
        ancestor_suppression_strength: float = 1.0,
        node_feature_decoder: torch.nn.Module | None = None,
    ):
        super().__init__()

        # decoder is a sequence of GNN blocks. There must be at least one
        self.decoder = torch.nn.ModuleList(decoder)
        
        if len(self.decoder) == 0:
            raise ValueError("At least one GNN block must be provided.")
        
        # parent logits are determined by an MLP
        self.parent_logit_mlp = parent_logit_mlp
        hidden_dim = self.decoder[0].out_features
        mlp_in_dim = self.parent_logit_mlp.dims[0][0]
        self.in_dim = mlp_in_dim - 3 * hidden_dim
        if self.in_dim < 0:
            raise ValueError(
                f"Invalid parent_logit_mlp dims: expected input >= 3*hidden_dim={3*hidden_dim}, "
                f"but got {mlp_in_dim}."
            )

        # decoder initial state
        self.decoder_init = decoder_init
        # Determine expected input dim for decoder blocks
        expected_dim = self.decoder[0].in_features
        mlp_in_dim = self.parent_logit_mlp.dims[0][0]
        inferred_latent_dim = mlp_in_dim - 3 * hidden_dim
        # If decoder_init is None, require latent_dim == hidden_dim
        if self.decoder_init is None:
            latent_dim = z_dim = inferred_latent_dim
            if latent_dim != hidden_dim:
                raise ValueError(
                    "decoder_init=None requires latent_dim == hidden_dim."
                )
        # Check initialization
        if self.decoder_init is not None:
            test = self.decoder_init(torch.zeros(1, self.decoder_init.dims[0][0]))
            if test.shape[-1] != expected_dim:
                raise ValueError(
                f"decoder_init produces {test.shape[-1]} elements, but decoder expects {expected_dim} elements."
                )

        # ancestor suppression strength to impose transitive reduction
        self.ancestor_suppression_strength = ancestor_suppression_strength
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
        self.node_feature_decoder = node_feature_decoder


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
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Unified forward() entrypoint for the decoder.
        Delegates to autoregressive_decode(), which implements the CST-style
        node-by-node graph generation.
        """
        return self.autoregressive_decode(z, teacher_forcing_targets, atom_count) # may add other methods later

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

        # latent -> initial state for node 0
        if z.dim() == 2 and z.size(0) == 1:
            z = z.squeeze(0)
            
        # Use decoder's init_state logic
        h0 = self.init_state(z)

        # initial structures
        node_states = [h0]                      # list of tensors
        # maintain dynamic edge index
        edge_index_list = []
        links = []                                # will append rows as lists of 0/1
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

        # dynamic positional embedding based on N_max
        hidden_dim = self.decoder[0].out_features
        self.positional_emb = torch.nn.Embedding(N_max, hidden_dim).to(h0.device)

        ancestors = []
        if self.ancestor_suppression:
            bit0 = torch.zeros(N_max, dtype=torch.uint8, device=h0.device)
            ancestors.append(bit0.clone())           # node 0 has no ancestors

        # adjacency row for node 0
        links.append([0])
        # accumulate per-step log likelihoods during training
        step_logprobs = []

        # grow nodes 1 .. N_max-1
        for t in range(1, N_max):
            # Insert provisional state for new node with positional embedding
            pos_e = self.positional_emb(torch.tensor(t, device=h0.device))
            node_states.append(h0 + pos_e)

            prev_count = t

            # GNN update BEFORE computing parent probabilities (state reflects graph up to t−1)
            if len(edge_index_list) > 0:
                edge_index_tensor = torch.tensor(edge_index_list, dtype=torch.long, device=h0.device).t().contiguous()
            else:
                edge_index_tensor = torch.empty((2,0), dtype=torch.long, device=h0.device)

            H = torch.stack(node_states, dim=0)
            for dec_layer in self.decoder:
                H = dec_layer(H, edge_index_tensor)

            # Update node_states and extract provisional state for node t
            node_states = [H[i] for i in range(t+1)]
            h_t = node_states[t]

            # compute parent logits for all previous nodes
            # Prepare inputs for parent_logit_mlp
            h_prev = torch.stack(node_states, dim=0)
            h_t_rep = h_t.unsqueeze(0).repeat(prev_count, 1)
            z_rep = z.unsqueeze(0).repeat(prev_count, 1)
            pos_t = self.positional_emb(torch.tensor(t, device=h0.device))
            pos_t_rep = pos_t.unsqueeze(0).repeat(prev_count, 1)
            parent_mlp_in = torch.cat([h_t_rep, h_prev[:prev_count], z_rep, pos_t_rep], dim=1)
            logits = self.parent_logit_mlp(parent_mlp_in).squeeze(-1)
            probs = torch.sigmoid(logits)

            if self.ancestor_suppression:
                # multiplicative suppression based on ancestor relations (S_i = product_{j>i}(1 - Anc[j][i]))
                suppression = torch.ones(prev_count, device=h0.device)
                for j in reversed(range(prev_count)):
                    anc_j = ancestors[j][:prev_count].float()
                    suppression *= (1.0 - self.ancestor_suppression_strength * anc_j)

                # final probabilities
                adjusted_probs = probs * suppression
            else:
                adjusted_probs = probs

            # teacher forcing vs. sampling
            if self.training and teacher_forcing_targets is not None:
                # ground truth row for node t: shape (prev_count,)
                gt_row = teacher_forcing_targets[t][:prev_count].to(h0.device)
                parent_mask = gt_row.to(torch.uint8)

                # accumulate log likelihood for this step
                eps = 1e-12
                log_p = (
                    gt_row * torch.log(adjusted_probs + eps)
                    + (1 - gt_row) * torch.log(1 - adjusted_probs + eps)
                )
                step_logprobs.append(log_p.sum())
            else:
                parent_mask = torch.bernoulli(adjusted_probs).to(torch.uint8)

            # build edges to new node t
            new_edges = []
            for i in range(prev_count):
                if parent_mask[i] == 1:
                    new_edges.append([i, t])

            # update adjacency structure
            row_t = [int(parent_mask[i].item()) for i in range(prev_count)]
            row_t.append(0)
            links.append(row_t)

            # update edge list
            for e in new_edges:
                edge_index_list.append(e)

            # update ancestors[t]: OR of ancestors of parents + parents themselves
            if self.ancestor_suppression:
                anc_t = torch.zeros(N_max, dtype=torch.uint8, device=h0.device)
                parent_indices = (parent_mask == 1).nonzero(as_tuple=False).flatten()
                for p in parent_indices.tolist():
                    anc_p = ancestors[p]
                    anc_t |= anc_p
                    anc_t[p] = 1
                ancestors.append(anc_t)


        # convert link matrix + node_states to tensors
        L = torch.tensor(links, dtype=torch.float32, device=h0.device)
        if self.node_feature_decoder is not None:
            # FINAL GNN UPDATE: ensure node_states reflect the full graph including last-node parents
            if len(edge_index_list) > 0:
                edge_index_tensor = torch.tensor(edge_index_list, dtype=torch.long, device=h0.device).t().contiguous()
            else:
                edge_index_tensor = torch.empty((2,0), dtype=torch.long, device=h0.device)

            H = torch.stack(node_states, dim=0)
            for dec_layer in self.decoder:
                H = dec_layer(H, edge_index_tensor)

            node_states = [H[i] for i in range(N_max)]

        X_hidden = torch.stack(node_states, dim=0)
        X_out = self.node_feature_decoder(X_hidden) if self.node_feature_decoder is not None else None
        # Determine whether to compute log-likelihood
        total_logprob = (
            torch.stack(step_logprobs).sum() if (teacher_forcing_targets is not None) else None
        )

        return L, X_out, total_logprob

    def reconstruct_link_matrix(self, z, atom_count):
        L, _, _ = self.decoder(z, atom_count=atom_count)
        return L

    def reconstruct_node_features(self, z, atom_count):
        L, X_out, _ = self.decoder(z, atom_count=atom_count)
        if X_out is None:
            raise RuntimeError("Decoder not configured with node_feature_decoder.")
        return L, X_out

    def compute_normalized_log_likelihood(self, z, teacher_forcing_targets):
        atom_count = teacher_forcing_targets.size(0)
        _, _, logprob = self.decoder(z, teacher_forcing_targets=teacher_forcing_targets)
        return logprob / (atom_count * (atom_count - 1) / 2)

    def full_decode(self, z, atom_count=None, teacher_forcing_targets=None):
        return self.decoder(z, atom_count=atom_count, teacher_forcing_targets=teacher_forcing_targets)
    
    @classmethod
    def from_config(cls, cfg):
        decoder = [
            QGGNN.GNNBlock.from_config(bcfg) for bcfg in cfg.get("decoder", [])
        ]
        parent_cfg = cfg["parent_logits"]
        parent_logit_mlp = QGLS.LinearSequential.from_config(parent_cfg)
        decoder_init_cfg = cfg.get("decoder_init", None)
        decoder_init = (
            QGLS.LinearSequential.from_config(decoder_init_cfg)
            if decoder_init_cfg is not None else None
        )
        node_feature_decoder_cfg = cfg.get("node_feature_decoder", None)
        node_feature_decoder = (
            QGLS.LinearSequential.from_config(node_feature_decoder_cfg)
            if node_feature_decoder_cfg is not None else None
        )
        return cls(
            decoder=decoder,
            parent_logit_mlp=parent_logit_mlp,
            decoder_init=decoder_init,
            ancestor_suppression_strength=cfg.get("ancestor_suppression_strength", 1.0),
            node_feature_decoder=node_feature_decoder,
        )

    def to_config(self):
        return {
            "decoder": [blk.to_config() for blk in self.decoder],
            "parent_logits": {
                "dims": self.parent_logit_mlp.dims,
                "activations": self.parent_logit_mlp.activations,
                "linear_kwargs": self.parent_logit_mlp.linear_kwargs,
                "activation_kwargs": self.parent_logit_mlp.activation_kwargs,
            },
            "decoder_init": None if self.decoder_init is None else {
                "dims": self.decoder_init.dims,
                "activations": self.decoder_init.activations,
                "linear_kwargs": self.decoder_init.linear_kwargs,
                "activation_kwargs": self.decoder_init.activation_kwargs,
            },
            "ancestor_suppression_strength": self.ancestor_suppression_strength,
            "node_feature_decoder": None if self.node_feature_decoder is None else {
                "dims": self.node_feature_decoder.dims,
                "activations": self.node_feature_decoder.activations,
                "linear_kwargs": self.node_feature_decoder.linear_kwargs,
                "activation_kwargs": self.node_feature_decoder.activation_kwargs,
            },
        }


class LatentModule(torch.nn.Module):
    """
    Encapsulates VAE latent bottleneck, mu/logvar heads, and latent_dim.
    """

    def __init__(
        self,
        bottleneck: QGLS.LinearSequential | None = None,
        mu_head: QGLS.LinearSequential | None = None,
        logvar_head: QGLS.LinearSequential | None = None,
    ) -> None:
        """
        Args:
            bottleneck (LinearSequential | None): Optional MLP applied before mu/logvar heads.
            mu_head (LinearSequential | None): LinearSequential producing the latent mean.
            logvar_head (LinearSequential | None): LinearSequential producing the latent log-variance.

        Returns:
            None
        """
        super().__init__()

        self.bottleneck = bottleneck
        self.mu_head = mu_head
        self.logvar_head = logvar_head

        # Consistency checks: only if both heads exist (VAE mode)
        if self.mu_head is not None and self.logvar_head is not None:
            # Input dims must match
            mu_in = self.mu_head.dims[0][0]
            logvar_in = self.logvar_head.dims[0][0]
            if mu_in != logvar_in:
                raise ValueError(
                    f"mu_head input dim ({mu_in}) != logvar_head input dim ({logvar_in}). They must match."
                )

            # Bottleneck output must match input dim of heads
            if self.bottleneck is not None:
                bottleneck_out = self.bottleneck.dims[-1][1]
                if bottleneck_out != mu_in:
                    raise ValueError(
                        f"Bottleneck output dim ({bottleneck_out}) must equal mu/logvar input dim ({mu_in})."
                    )

            # Output dims must match
            mu_last = self.mu_head.layers[-1]
            logvar_last = self.logvar_head.layers[-1]

            if not (hasattr(mu_last, "out_features") and hasattr(logvar_last, "out_features")):
                raise ValueError("Cannot infer latent output dimension: mu/logvar head lacks .out_features.")

            if mu_last.out_features != logvar_last.out_features:
                raise ValueError(
                    f"mu_head output dim ({mu_last.out_features}) != "
                    f"logvar_head output dim ({logvar_last.out_features})."
                )

            # Assign latent dimension
            self.lat_dim = mu_last.out_features

        else:
            # Non‑VAE mode → latent dim determined later by encoder
            self.lat_dim = None

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Args:
            h (torch.Tensor): Input tensor representing the graph-level embedding.

        Returns:
            tuple:
                - z (torch.Tensor): Sampled latent vector (or h if no VAE heads are defined).
                - mu (torch.Tensor | None): Mean vector of the latent distribution, or None.
                - logvar (torch.Tensor | None): Log-variance vector of the latent distribution, or None.
        """
        if self.mu_head is None or self.logvar_head is None:
            return h, None, None
        if self.bottleneck is not None:
            h = self.bottleneck(h)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

    def to_config(self) -> dict[str, Any]:
        def head_cfg(head):
            if head is None:
                return None
            return {
                "dims": head.dims,
                "activations": head.activations,
                "linear_kwargs": head.linear_kwargs,
                "activation_kwargs": head.activation_kwargs,
            }
        return {
            "bottleneck": head_cfg(self.bottleneck),
            "mu_head": head_cfg(self.mu_head),
            "logvar_head": head_cfg(self.logvar_head),
            "dim": (self.mu_head.layers[-1].out_features if self.mu_head is not None else None),
        }

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "LatentModule":
        bottleneck = QGLS.LinearSequential.from_config(cfg["bottleneck"]) if cfg.get("bottleneck") else None
        mu_head = QGLS.LinearSequential.from_config(cfg["mu_head"]) if cfg.get("mu_head") else None
        logvar_head = QGLS.LinearSequential.from_config(cfg["logvar_head"]) if cfg.get("logvar_head") else None
        return cls(bottleneck=bottleneck, mu_head=mu_head, logvar_head=logvar_head)

class GNNAutoEncoder(torch.nn.Module):
    """
    Pure graph autoencoder:
      - encoder: EncoderModule instance
      - decoder: a DecoderModule instance
      - latent: a LatentModule instance
    """

    def __init__(
        self,
        encoder: EncoderModule,
        decoder: DecoderModule,
        latent: LatentModule,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent = latent

        # latent dimension from latent module
        lat_dim = self.latent.lat_dim

        # encoder output dimension
        enc_dim = self.encoder.out_dim

        # decoder input dimension
        dec_dim = self.decoder.in_dim

        # VAE mode
        if self.latent.mu_head is not None:
            if lat_dim != dec_dim:
                raise ValueError(
                    f"Inconsistent latent dims: latent outputs {lat_dim} but decoder expects {dec_dim}."
                )
            elif enc_dim != lat_dim:
                raise ValueError(
                    f"Inconsistent latent dims: encoder outputs {enc_dim} but latent expects {lat_dim}."
                )
            self.latent_dim = lat_dim

        # non‑VAE mode
        else:
            # latent_dim is enc_dim
            if enc_dim != dec_dim:
                raise ValueError(
                    f"Non‑VAE mode: encoder out_dim={enc_dim} must equal decoder latent_in_dim={dec_dim}."
                )
            self.latent_dim = enc_dim
        

    def sample(self, N: int, z: torch.Tensor | None = None) -> tuple:
        if z is None:
            if self.latent_dim is None:
                raise ValueError("Cannot sample z without latent_dim.")
            z = torch.randn(self.latent_dim, device=next(self.parameters()).device)
        self.eval()
        return self.decode(z, atom_count=N)
    
    # sample_latent method removed; handled by latent_module

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        gcn_kwargs: dict | None = None,
        graph_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encoder(x, edge_index, batch, gcn_kwargs=gcn_kwargs, graph_features=graph_features)

    def encode_to_latent(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        gcn_kwargs: dict | None = None,
        graph_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Convenience helper: run encoder + latent to obtain the latent vector z.
        This bypasses decoding and returns only (z, mu, logvar).
        """
        # Encode to graph-level embedding
        h = self.encoder(x, edge_index, batch, gcn_kwargs=gcn_kwargs, graph_features=graph_features)
        # Apply latent module
        z, mu, logvar = self.latent(h)
        return z, mu, logvar

    def decode(
        self,
        z: torch.Tensor,
        atom_count: int | None = None,
        teacher_forcing_targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        General decoder interface:
          - z: latent vector
          - atom_count: number of nodes to generate
          - teacher_forcing_targets: optional adjacency matrix for teacher forcing
        """
        # Allow shape [latent_dim] or [1, latent_dim]
        if z.dim() == 2 and z.size(0) == 1:
            z = z.squeeze(0)
        elif z.dim() != 1:
            raise ValueError(f"decode() expects shape [latent_dim] or [1, latent_dim], got {tuple(z.shape)}")
        return self.decoder(
            z,
            teacher_forcing_targets=teacher_forcing_targets,
            atom_count=atom_count,
        )



    def forward(
        self,
        x: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        graph_features: torch.Tensor | None = None,
        teacher_forcing_targets: torch.Tensor | None = None,
        atom_count: int | None = None,
        latent_override: torch.Tensor | None = None,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None],
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:

        # Case 1: latent override → skip encoder entirely
        if latent_override is not None:
            z = latent_override
            mu = None
            logvar = None
            if z.dim() != 1 or z.size(0) != self.latent_dim:
                raise ValueError(f"latent_override must be shape [{self.latent_dim}], got {tuple(z.shape)}")
        else:
            # Case 2: normal VAE encode → sample latent
            h = self.encode(x, edge_index, batch)

        # Optional graph feature aggregation
        if graph_features is not None and self.encoder.graph_features_net is not None:
            graph_features_proc = self.encoder.graph_features_net(graph_features)
        else:
            graph_features_proc = None

        if self.encoder.aggregate_graph_features is not None and graph_features_proc is not None:
            if latent_override is None:
                h = self.encoder.aggregate_graph_features(h, graph_features_proc)

        # Sample latent
        if latent_override is None:
            z, mu, logvar = self.latent(h)
        else:
            # latent_override bypasses VAE sampling
            mu = None
            logvar = None

        # Decode (autoregressive only)
        adj_hat, x_hat, logprob = self.decoder(
            z,
            teacher_forcing_targets=teacher_forcing_targets,
            atom_count=atom_count,
        )

        return (adj_hat, x_hat, logprob), z, mu, logvar
    
    
    def to_config(self) -> dict[str, Any]:
        """Serialize the autoencoder to a config dictionary."""
        config = {
            "encoder": self.encoder.to_config(),
            "decoder": self.decoder.to_config(),
            "latent": self.latent.to_config(),
        }
        return config

    def save(self, path: str | Path) -> None:
        """Save the autoencoder state to file, including config and state_dict."""
        config = self.to_config()
        torch.save(
            {"config": config, "model": self.state_dict()},
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: torch.device = torch.device("cpu"),
    ) -> "GNNAutoEncoder":
        model_dict = torch.load(path, weights_only=False)
        model = cls.from_config(model_dict["config"]).to(device)
        model.load_state_dict(model_dict["model"])
        return model

    @classmethod
    def _cfg_helper(
        cls,
        cfg: dict,
        utility_function: Callable,
        throw_message: str,
    ) -> torch.nn.Module | Callable:
        """Helper function to create a module or callable from a config."""
        if not utils.verify_config_node(cfg):
            raise ValueError(throw_message)
        f = utility_function(cfg["type"])
        if f is None:
            raise ValueError(
                f"Utility function '{utility_function.__name__}' could not find '{cfg['type']}'"
            )
        if isinstance(f, type):
            return f(
                cfg["args"] if "args" in cfg else [],
                **(cfg["kwargs"] if "kwargs" in cfg else {}),
            )
        else:
            return f

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GNNAutoEncoder":
        encoder = EncoderModule.from_config(config["encoder"])
        decoder = DecoderModule.from_config(config["decoder"])
        latent = LatentModule.from_config(config.get("latent", {}))
        return cls(
            encoder=encoder,
            decoder=decoder,
            latent=latent,
        )
