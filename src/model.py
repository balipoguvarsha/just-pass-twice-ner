"""
Model architectures for:
1. BaselineTokenClassifier: Standard causal single-pass model with linear classification head.
2. JustPassTwiceNER: Paper-inspired causal model with Two-Pass contextualization and Definition-Guided Bilinear Scorer.
3. MultiDefJustPassTwiceNER: Proposed improvement aggregating multiple paraphrased entity definitions.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F



class ProjectionMLP(nn.Module):
    """2-layer MLP with LayerNorm and GELU as specified in the JPT paper."""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = out_dim * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaselineTokenClassifier(nn.Module):
    """
    Standard supervised token classification baseline.
    Single forward pass with standard linear projection head.
    Ablates both Two-Pass bidirectional contextualization and definition embeddings.
    """
    def __init__(
        self,
        backbone: nn.Module,
        num_labels: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.backbone = backbone
        hidden_size = backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # [batch_size, seq_len, num_labels]

        loss = None
        if labels is not None:
            # Flatten tokens
            active_loss = labels.view(-1) != -100
            active_logits = logits.view(-1, logits.size(-1))[active_loss]
            active_labels = labels.view(-1)[active_loss]
            if active_labels.numel() > 0:
                loss = F.cross_entropy(active_logits, active_labels, weight=class_weights)
            else:
                loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)

        return {
            "loss": loss,
            "logits": logits
        }


class JustPassTwiceNER(nn.Module):
    """
    Paper-inspired Just Pass Twice (JPT) Model:
    1. Causal LM processes duplicated sequence: [Pass1, SEP, Pass2]
    2. Hidden representations extracted solely from Pass 2 (bidirectional context)
    3. Token representation projected: t_i = MLP_tok(h_i)
    4. Entity definition projected: p_j = MLP_ent(e_j)
    5. Bilinear classification score: S_{i, j} = t_i^T W p_j + b_j
    """
    def __init__(
        self,
        backbone: nn.Module,
        projection_dim: int = 128,
        entity_emb_dim: int = 128,
        num_entity_types: int = 5,
        dropout: float = 0.1,
        o_class_weight: float = 0.25
    ):
        super().__init__()
        self.backbone = backbone
        hidden_size = backbone.config.hidden_size
        self.projection_dim = projection_dim
        self.num_entity_types = num_entity_types
        self.o_class_weight = o_class_weight

        # Projections to shared semantic space R^{d_p}
        self.token_proj = ProjectionMLP(hidden_size, projection_dim, dropout=dropout)
        self.entity_proj = ProjectionMLP(entity_emb_dim, projection_dim, dropout=dropout)

        # Bilinear interaction matrix: W in R^{d_p x d_p}
        self.bilinear_w = nn.Parameter(torch.empty(projection_dim, projection_dim))
        nn.init.xavier_uniform_(self.bilinear_w)

        # Type-specific bias: b in R^{num_entity_types}
        self.bias = nn.Parameter(torch.zeros(num_entity_types))

    def compute_bilinear_scores(
        self,
        projected_tokens: torch.Tensor,   # [batch_size, seq_len, d_p]
        projected_entities: torch.Tensor  # [num_types, d_p]
    ) -> torch.Tensor:
        """
        S_{i, j} = t_i^T W p_j + b_j
        """
        # Intermediate: tokens @ W -> [batch_size, seq_len, d_p]
        tok_w = torch.matmul(projected_tokens, self.bilinear_w)
        # Compatibility with entity definitions: [batch_size, seq_len, num_types]
        scores = torch.matmul(tok_w, projected_entities.t())
        if self.bias.size(0) == projected_entities.size(0):
            scores = scores + self.bias
        else:
            # Dynamic bias handling for zero-shot cross-domain transfer with unseen entity counts
            ent_bias = self.bias[1:].mean() if self.bias.size(0) > 1 else self.bias[0]
            dyn_bias = torch.full((projected_entities.size(0),), ent_bias, device=projected_entities.device)
            dyn_bias[0] = self.bias[0]
            scores = scores + dyn_bias
        return scores

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        entity_embeddings: torch.Tensor,  # [num_types, entity_emb_dim]
        labels: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # Project tokens to shared space
        proj_tokens = self.token_proj(last_hidden)  # [batch_size, seq_len, d_p]
        # Project entity definitions to shared space
        proj_entities = self.entity_proj(entity_embeddings)  # [num_types, d_p]

        # Bilinear interaction
        logits = self.compute_bilinear_scores(proj_tokens, proj_entities)

        loss = None
        if labels is not None:
            active_loss = labels.view(-1) != -100
            active_logits = logits.view(-1, logits.size(-1))[active_loss]
            active_labels = labels.view(-1)[active_loss]

            if active_labels.numel() > 0:
                if class_weights is None:
                    # Default paper weight: downweight O class
                    weights = torch.ones(self.num_entity_types, device=input_ids.device)
                    weights[0] = self.o_class_weight  # Index 0 is 'O'
                else:
                    weights = class_weights
                loss = F.cross_entropy(active_logits, active_labels, weight=weights)
            else:
                loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)

        return {
            "loss": loss,
            "logits": logits,
            "projected_tokens": proj_tokens,
            "projected_entities": proj_entities
        }


class MultiDefJustPassTwiceNER(JustPassTwiceNER):
    """
    Proposed Improvement:
    Aggregates multiple paraphrased definitions per entity type.
    Input entity_embeddings has shape: [num_types, num_paraphrases, entity_emb_dim]
    Definitions are independently projected and pooled (mean-pooled or attention-pooled)
    to form a semantically richer, more robust entity representation:
    p_j = mean_m(MLP_ent(e_{j, m}))
    """
    def __init__(
        self,
        backbone: nn.Module,
        projection_dim: int = 128,
        entity_emb_dim: int = 128,
        num_entity_types: int = 5,
        dropout: float = 0.1,
        o_class_weight: float = 0.25,
        aggregation: str = "mean"
    ):
        super().__init__(
            backbone=backbone,
            projection_dim=projection_dim,
            entity_emb_dim=entity_emb_dim,
            num_entity_types=num_entity_types,
            dropout=dropout,
            o_class_weight=o_class_weight
        )
        self.aggregation = aggregation
        if aggregation == "attention":
            self.attn_weights = nn.Linear(projection_dim, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        entity_embeddings: torch.Tensor,  # [num_types, num_paraphrases, entity_emb_dim] or [num_types, entity_emb_dim]
        labels: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state

        proj_tokens = self.token_proj(last_hidden)  # [batch_size, seq_len, d_p]

        if entity_embeddings.dim() == 3:
            num_types, num_paraphrases, emb_dim = entity_embeddings.shape
            flat_embs = entity_embeddings.view(num_types * num_paraphrases, emb_dim)
            proj_flat = self.entity_proj(flat_embs)
            proj_multi = proj_flat.view(num_types, num_paraphrases, self.projection_dim)

            if self.aggregation == "mean":
                proj_entities = proj_multi.mean(dim=1)  # [num_types, d_p]
            elif self.aggregation == "attention":
                attn_logits = self.attn_weights(proj_multi).squeeze(-1)  # [num_types, num_paraphrases]
                attn_probs = F.softmax(attn_logits, dim=-1).unsqueeze(-1)  # [num_types, num_paraphrases, 1]
                proj_entities = (proj_multi * attn_probs).sum(dim=1)
            else:
                proj_entities = proj_multi.mean(dim=1)
        else:
            proj_entities = self.entity_proj(entity_embeddings)

        logits = self.compute_bilinear_scores(proj_tokens, proj_entities)

        loss = None
        if labels is not None:
            active_loss = labels.view(-1) != -100
            active_logits = logits.view(-1, logits.size(-1))[active_loss]
            active_labels = labels.view(-1)[active_loss]

            if active_labels.numel() > 0:
                if class_weights is None:
                    weights = torch.ones(self.num_entity_types, device=input_ids.device)
                    weights[0] = self.o_class_weight
                else:
                    weights = class_weights
                loss = F.cross_entropy(active_logits, active_labels, weight=weights)
            else:
                loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)

        return {
            "loss": loss,
            "logits": logits,
            "projected_tokens": proj_tokens,
            "projected_entities": proj_entities
        }
