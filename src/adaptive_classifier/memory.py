import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import faiss
import logging
from .models import Example, ModelConfig

logger = logging.getLogger(__name__)

class PrototypeMemory:
    """Memory system that maintains prototypes for each class."""
    
    def __init__(
        self,
        embedding_dim: int,
        config: Optional[ModelConfig] = None
    ):
        """Initialize the prototype memory system.
        
        Args:
            embedding_dim: Dimension of the embeddings
            config: Optional model configuration
        """
        self.embedding_dim = embedding_dim
        self.config = config or ModelConfig()
        
        # Initialize storage
        self.examples = defaultdict(list)  # label -> List[Example]
        self.prototypes = {}  # label -> tensor
        self.strategic_prototypes = {}  # label -> strategic prototype tensor
        # Class spread recorded when a classifier was saved. A reloaded memory
        # keeps only a few representative examples, which understate the spread.
        self.saved_radii = {}  # label -> float
        # A reloaded memory holds only a few examples per class, but its prototypes were
        # averaged over many more. For such classes the prototype keeps that average
        # (prototype_base: label -> (prototype, number of examples behind it)) and folds in
        # examples added since (post_sum / post_n) instead of re-averaging the few that
        # were kept, which would jerk the prototype away from where it was saved.
        self.prototype_base = {}  # label -> (tensor, int)
        self.post_sum = {}        # label -> tensor
        self.post_n = {}          # label -> int
        
        # Initialize FAISS index for fast similarity search
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.label_to_index = {}  # label -> index in FAISS
        self.index_to_label = {}  # index in FAISS -> label
        
        # Statistics
        self.updates_since_rebuild = 0
    
    def add_example(self, example: Example, label: str):
        """Add a new example to memory.
        
        Args:
            example: Example to add
            label: Class label
            
        Raises:
            ValueError: If example embedding dimension doesn't match memory dimension
        """
        # Validate embedding dimension
        if example.embedding is None:
            raise ValueError("Example must have an embedding")
        if example.embedding.size(-1) != self.embedding_dim:
            raise ValueError(
                f"Example embedding dimension {example.embedding.size(-1)} "
                f"does not match memory dimension {self.embedding_dim}"
            )
            
        # Keep memory on the CPU: FAISS indexes and saved state are CPU-only, and
        # mixing devices across examples would break torch.stack in prototype updates.
        example.embedding = example.embedding.detach().cpu()

        # Add new example
        self.examples[label].append(example)
        if label in self.prototype_base:
            self.post_sum[label] = self.post_sum[label] + example.embedding
            self.post_n[label] += 1
        
        # Keep at most max_examples_per_class (newest plus boundary examples; see _prune_examples).
        if len(self.examples[label]) > self.config.max_examples_per_class:
            self._prune_examples(label)
        
        # Update prototype
        self._update_prototype(label)

        # Only increment if we haven't just rebuilt
        if not getattr(self, 'just_rebuilt', False):
            self.updates_since_rebuild += 1
            # print(f"updates_since_rebuild: {self.updates_since_rebuild}")
        
        # If we've hit the frequency, rebuild
        if self.updates_since_rebuild >= self.config.prototype_update_frequency:
            # print("Index rebuild")
            self._rebuild_index()
            self.just_rebuilt = True
        else:
            self.just_rebuilt = False
            
        # print(f"updates_since_rebuild: {self.updates_since_rebuild}")
    
    def get_nearest_prototypes(
            self,
            query_embedding: torch.Tensor,
            k: int = 5,
            min_similarity: Optional[float] = None
        ) -> List[Tuple[str, float]]:
            """Find the nearest prototype neighbors for a query.
            
            Args:
                query_embedding: Query embedding tensor
                k: Number of neighbors to return
                min_similarity: Optional minimum similarity threshold
                
            Returns:
                List of (label, similarity) tuples
            """
            # Ensure index is up to date
            if self.updates_since_rebuild >= self.config.prototype_update_frequency:
                self._rebuild_index()
                
            # Handle empty index case
            if self.index.ntotal == 0:
                return []
                
            # Ensure the query is in the right format
            query_np = query_embedding.unsqueeze(0).numpy()
                
            # Search the index with valid k
            k = min(k, self.index.ntotal)
            distances, indices = self.index.search(query_np, k)
            
            temperature = getattr(self.config, 'prototype_temperature', None)
            valid = [(int(i), float(d)) for i, d in zip(indices[0], distances[0]) if i >= 0]
            if not valid:
                return []

            if temperature:
                # Softmax over negative squared distance: the nearest class gets
                # most of the mass when it is clearly nearer than the others.
                scores = torch.softmax(
                    torch.tensor([-d / temperature for _, d in valid]), dim=0
                )
            else:
                # Legacy scoring (before 0.3.0): exp(-distance), then softmax.
                scores = torch.softmax(
                    torch.tensor([float(np.exp(-d)) for _, d in valid]), dim=0
                )

            results = [
                (self.index_to_label[i], float(score))
                for (i, _), score in zip(valid, scores)
            ]
            return results
    
    def _update_prototype(self, label: str):
        """Update the prototype for a given label.
        
        Args:
            label: Class label to update
        """
        examples = self.examples[label]
        if not examples:
            return
            
        if label in self.prototype_base:
            # The class holds at most max_examples_per_class examples: the post_n added
            # since loading, plus as many of the older ones (summarised by the saved
            # prototype) as still fit under that cap. The first eviction drops the base.
            base, count = self.prototype_base[label]
            post_n = self.post_n[label]
            weight = max(0, min(count, self.config.max_examples_per_class - post_n))
            prototype = (base * weight + self.post_sum[label]) / (weight + post_n)
        else:
            # Compute mean of embeddings
            embeddings = torch.stack([ex.embedding for ex in examples])
            prototype = torch.mean(embeddings, dim=0)
        
        # Update prototype
        self.prototypes[label] = prototype
        
        # Update index if label exists
        if label in self.label_to_index:
            idx = self.label_to_index[label]
            self.index.remove_ids(torch.tensor([idx]))
            self.index.add(prototype.unsqueeze(0).numpy())
    
    def _rebuild_index(self):
        """Rebuild the FAISS index from scratch."""
        # Clear existing index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.label_to_index.clear()
        self.index_to_label.clear()
        
        # Add all prototypes in sorted order to ensure consistent indices
        sorted_labels = sorted(self.prototypes.keys())
        for i, label in enumerate(sorted_labels):
            prototype = self.prototypes[label]
            self.index.add(prototype.unsqueeze(0).numpy())
            self.label_to_index[label] = i
            self.index_to_label[i] = label
            
        # Explicitly reset the counter
        self.updates_since_rebuild = 0

    def _restore_from_save(self):
        """Restore index and mappings after loading from save."""
        # Clear existing index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.label_to_index.clear()
        self.index_to_label.clear()
        
        # Add prototypes in sorted order for consistency
        sorted_labels = sorted(self.prototypes.keys())
        for i, label in enumerate(sorted_labels):
            prototype = self.prototypes[label]
            self.index.add(prototype.unsqueeze(0).numpy())
            self.label_to_index[label] = i
            self.index_to_label[i] = label
            
        self.updates_since_rebuild = 0
    
    def _prune_examples(self, label: str):
        """Evict examples of `label` until at most max_examples_per_class remain.

        Retention policy, "recent + boundary": examples are held in the order
        they were added (a reloaded memory's saved examples count as the
        oldest). The newest `ceil(retention_recent_fraction * limit)` are always
        kept, so the class follows drift. The rest of the cap goes to the older
        examples that lie closest to another class's prototype: the borderline
        cases that decide the boundary and that give the class its true spread.
        With no other class, the remaining slots also go to the newest examples.
        Kept examples stay in arrival order.

        The caller recomputes the prototype (now the mean of the held examples)
        and the search index; the neural head is retrained on what remains by
        the next training pass (`add_examples` always runs one).
        """
        limit = self.config.max_examples_per_class
        examples = self.examples[label]
        if len(examples) <= limit:
            return 0
        evicted = len(examples) - limit
        fraction = float(getattr(self.config, 'retention_recent_fraction', 0.5))
        n_recent = min(limit, math.ceil(fraction * limit))
        n_boundary = limit - n_recent
        split = len(examples) - n_recent
        older, recent = examples[:split], examples[split:]
        others = [proto for other, proto in self.prototypes.items() if other != label]
        if n_boundary == 0:
            kept_older = []
        elif not others:
            kept_older = older[len(older) - n_boundary:]
        else:
            embeddings = torch.stack([ex.embedding for ex in older]).float()
            to_other = torch.cdist(embeddings, torch.stack(others).float()).min(dim=1).values
            nearest = np.argsort(to_other.numpy(), kind='stable')[:n_boundary]
            kept_older = [older[i] for i in sorted(int(i) for i in nearest)]
        self.examples[label] = kept_older + recent
        # The saved prototype and spread summarised examples that were never held,
        # and an evicted set can no longer be described by them: from now on the
        # class is described by the examples it holds.
        if label in self.prototype_base:
            self._forget_base(label)
            self.saved_radii.pop(label, None)
        return evicted

    def enforce_limit(self) -> int:
        """Apply max_examples_per_class to every class (after lowering it). Returns the number evicted."""
        evicted = 0
        for label in list(self.examples):
            removed = self._prune_examples(label)
            if removed:
                evicted += removed
                self._update_prototype(label)
        if evicted:
            self._rebuild_index()
        return evicted
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics.
        
        Returns:
            Dictionary of memory statistics
        """
        return {
            'num_classes': len(self.prototypes),
            'examples_per_class': {
                label: len(examples)
                for label, examples in self.examples.items()
            },
            'total_examples': sum(
                len(examples) for examples in self.examples.values()
            ),
            'prototype_dimensions': self.embedding_dim,
            'updates_since_rebuild': self.updates_since_rebuild
        }
    
    def set_prototype_base(self, label: str, count: int):
        """Treat the current prototype of `label` as the mean of `count` examples.

        Used after loading, when fewer examples than that are held in memory.
        """
        if label in self.prototypes and count > len(self.examples.get(label, [])):
            self.prototype_base[label] = (self.prototypes[label].clone(), int(count))
            self.post_sum[label] = torch.zeros_like(self.prototypes[label])
            self.post_n[label] = 0

    def _forget_base(self, label: str):
        self.prototype_base.pop(label, None)
        self.post_sum.pop(label, None)
        self.post_n.pop(label, None)

    def remove_class(self, label: str):
        """Drop a class and everything stored for it."""
        self._forget_base(label)
        self.examples.pop(label, None)
        self.prototypes.pop(label, None)
        self.strategic_prototypes.pop(label, None)
        self.saved_radii.pop(label, None)
        self._rebuild_index()

    def remove_examples(self, label: str, texts) -> int:
        """Remove examples of `label` whose text is in `texts`.

        The prototype is recomputed from what remains; a class left with no
        examples is removed. Returns the number of examples removed.
        """
        wanted = set(texts)
        before = self.examples.get(label, [])
        kept = [ex for ex in before if ex.text not in wanted]
        removed = len(before) - len(kept)
        if removed == 0:
            return 0
        if not kept:
            self.remove_class(label)
            return removed
        self.examples[label] = kept
        self._forget_base(label)       # the prototype is recomputed from what remains
        self.prototypes[label] = torch.mean(torch.stack([ex.embedding for ex in kept]), dim=0)
        self.strategic_prototypes.pop(label, None)
        # The saved spread described the old membership.
        self.saved_radii.pop(label, None)
        self._rebuild_index()
        return removed

    def class_radius(self, label: str) -> float:
        """The `ood_radius_percentile`-th percentile of example-to-prototype distances.

        Computed over the examples held for the class. A class whose memory
        holds only part of what its prototype was built from (a reloaded or
        merged classifier) also has a recorded radius, which acts as a floor.
        """
        radius = self.saved_radii.get(label, 0.0)
        examples = self.examples.get(label)
        if examples and label in self.prototypes:
            embeddings = torch.stack([ex.embedding for ex in examples])
            distances = torch.norm(embeddings - self.prototypes[label], dim=1).numpy()
            percentile = float(getattr(self.config, 'ood_radius_percentile', 95.0))
            radius = max(radius, float(np.percentile(distances, percentile)))
        return radius

    def ood_scores(self, query_embeddings: torch.Tensor):
        """Batched `ood_score`: (scores, labels) for each row of `query_embeddings`."""
        n = len(query_embeddings)
        if not self.prototypes:
            return [float('inf')] * n, [None] * n
        labels = list(self.prototypes)
        prototypes = torch.stack([self.prototypes[label] for label in labels]).float()
        radii = torch.tensor([max(self.class_radius(label), self.config.ood_min_radius)
                              for label in labels], dtype=torch.float32)
        queries = torch.as_tensor(query_embeddings, dtype=torch.float32).reshape(n, -1)
        ratios = torch.cdist(queries, prototypes) / radii
        best, index = ratios.min(dim=1)
        return [float(v) for v in best], [labels[int(i)] for i in index]

    def ood_score(self, query_embedding: torch.Tensor):
        """Return (score, label): the smallest distance-to-radius ratio over classes.

        1.0 means the query is exactly as far from the nearest prototype as that
        class's radius (by default the 95th percentile of its training examples'
        distances, so about 5% of its own training examples score above 1.0).
        Returns (inf, None) when the memory is empty.
        """
        scores, labels = self.ood_scores(query_embedding.reshape(1, -1))
        return scores[0], labels[0]

    def clear(self):
        """Clear all memory."""
        self.examples.clear()
        self.prototypes.clear()
        self.strategic_prototypes.clear()
        self.saved_radii.clear()
        self.prototype_base.clear()
        self.post_sum.clear()
        self.post_n.clear()
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.label_to_index.clear()
        self.index_to_label.clear()
        self.updates_since_rebuild = 0
    
    def compute_strategic_prototypes(self, cost_function, classifier_func):
        """Compute strategic prototypes for all classes.
        
        Args:
            cost_function: Strategic cost function
            classifier_func: Current classifier function
        """
        for label, examples in self.examples.items():
            if examples:
                strategic_embeddings = []
                
                for example in examples:
                    # Compute where this example would strategically move
                    strategic_embedding = cost_function.compute_best_response(
                        example.embedding, classifier_func
                    )
                    strategic_embeddings.append(strategic_embedding)
                
                # Compute mean of strategic embeddings
                if strategic_embeddings:
                    strategic_prototype = torch.stack(strategic_embeddings).mean(dim=0)
                    self.strategic_prototypes[label] = strategic_prototype
    
    def get_strategic_prototypes(self, query_embedding: torch.Tensor, k: int = 5) -> List[Tuple[str, float]]:
        """Get nearest strategic prototypes.
        
        Args:
            query_embedding: Query embedding tensor
            k: Number of neighbors to return
            
        Returns:
            List of (label, similarity) tuples
        """
        if not self.strategic_prototypes:
            return self.get_nearest_prototypes(query_embedding, k)
        
        # Compute similarities to strategic prototypes
        similarities = []
        for label, prototype in self.strategic_prototypes.items():
            # Compute cosine similarity
            sim = F.cosine_similarity(
                query_embedding.unsqueeze(0), 
                prototype.unsqueeze(0)
            ).item()
            similarities.append((label, sim))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]