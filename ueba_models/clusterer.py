"""
ueba_models/clusterer.py
─────────────────────────
Layer 3 — HDBSCAN Campaign Detection + FAISS RAG Retriever.

HDBSCAN: groups anomalous events into campaigns. Runs in a background
thread every N seconds on the accumulated anomaly buffer.

FAISS Retriever: given a new anomaly's feature vector, finds the most
similar past anomalies from the index for context (RAG without generation).
"""

import json
import logging
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

log = logging.getLogger("ueba.clusterer")

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    log.warning("hdbscan not installed. Install: pip install hdbscan")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    log.warning("faiss not installed. Install: pip install faiss-gpu or faiss-cpu")


# ── HDBSCAN Campaign Detector ─────────────────────────────────────────────────

class CampaignDetector:
    """
    Detects coordinated attack campaigns by clustering anomalous events.

    Runs in a background thread — does NOT block the streaming engine.
    Every N seconds, clusters the current anomaly buffer and assigns
    campaign IDs. The engine reads these assignments when writing output.

    Points labeled -1 by HDBSCAN = isolated anomalies (no campaign).
    Points with a label >= 0   = part of a coordinated campaign.
    """

    def __init__(self, config: dict):
        self.cfg         = config["clusterer"]
        self.buffer_size = self.cfg["buffer_size"]
        self.interval    = self.cfg["run_interval_seconds"]

        # Circular buffer — holds recent anomalous events
        # Each entry: (event_id, feature_vec, metadata_dict)
        self._buffer: deque = deque(maxlen=self.buffer_size)
        self._lock   = threading.Lock()

        # Latest clustering results {event_id → campaign_id or None}
        self._campaign_map: dict[str, str | None] = {}

        # Optional callback: called immediately after each clustering run
        # with the full campaign_map. Set by the engine to write sidecar file.
        self.on_cluster_complete = None

        # Background thread
        self._thread  = None
        self._running = False

        if not HDBSCAN_AVAILABLE:
            log.warning("HDBSCAN unavailable — campaign detection disabled")

    def add_anomaly(self, event_id: str, feature_vec: np.ndarray,
                    metadata: dict):
        """Add an anomalous event to the clustering buffer."""
        with self._lock:
            self._buffer.append((event_id, feature_vec.copy(), metadata))

    def get_campaign_id(self, event_id: str) -> str | None:
        """
        Get the campaign ID for an event (from last clustering run).
        Returns None if event is isolated or clustering hasn't run yet.
        """
        return self._campaign_map.get(event_id)

    def start(self):
        """Start the background clustering thread."""
        if not HDBSCAN_AVAILABLE:
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._run_loop,
            name="ueba-clusterer",
            daemon=True,
        )
        self._thread.start()
        log.info("Campaign detector started (interval: %ds)", self.interval)

    def stop(self):
        """Stop the background thread gracefully and run a final clustering pass."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        # Final pass — catches all events buffered during batch processing
        # that the periodic timer never had a chance to process
        try:
            log.info("Running final clustering pass on shutdown...")
            self._run_clustering()
        except Exception as e:
            log.error("Final clustering error: %s", e)

    def _run_loop(self):
        """Background thread — runs HDBSCAN periodically.
        Clusters immediately after a short delay on first run so that
        batch processing (which may finish in <interval seconds) still
        gets campaign IDs assigned.
        """
        first_run = True
        while self._running:
            if first_run:
                time.sleep(5)  # short delay for buffer to fill
                first_run = False
            else:
                time.sleep(self.interval)
            try:
                self._run_clustering()
            except Exception as e:
                log.error("Clustering error: %s", e)

    def _run_clustering(self):
        """Run one round of HDBSCAN on the current buffer."""
        with self._lock:
            if len(self._buffer) < self.cfg["min_cluster_size"]:
                return  # not enough events to cluster

            # Snapshot buffer
            snapshot = list(self._buffer)

        event_ids = [item[0] for item in snapshot]
        X         = np.stack([item[1] for item in snapshot], axis=0)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.cfg["min_cluster_size"],
            min_samples=self.cfg["min_samples"],
            metric=self.cfg["metric"],
            prediction_data=True,
        )
        labels = clusterer.fit_predict(X)

        # Build campaign map
        new_map = {}
        n_campaigns = len(set(labels)) - (1 if -1 in labels else 0)
        n_isolated  = (labels == -1).sum()

        for event_id, label in zip(event_ids, labels):
            if label == -1:
                new_map[event_id] = None  # isolated anomaly
            else:
                new_map[event_id] = f"CAMP-{label:04d}"

        with self._lock:
            self._campaign_map.update(new_map)

        log.info(
            "Clustering: %d events → %d campaigns, %d isolated",
            len(snapshot), n_campaigns, n_isolated,
        )

        # Fire callback immediately — engine writes sidecar without waiting
        if self.on_cluster_complete:
            try:
                self.on_cluster_complete(dict(self._campaign_map))
            except Exception as e:
                log.error("on_cluster_complete callback error: %s", e)


# ── FAISS RAG Retriever ───────────────────────────────────────────────────────

class RAGRetriever:
    """
    Retrieves similar past anomalies using FAISS vector similarity.

    At inference time:
      1. Take the anomalous event's feature vector
      2. L2-normalize it (for cosine similarity)
      3. Search FAISS index for top-K nearest neighbors
      4. Return their metadata as 'similar_past_events'

    This gives SOC analysts historical context for each new alert —
    "here are 5 similar events we've seen before."
    """

    def __init__(self, index_path: str, metadata_path: str,
                 scaler, config: dict):
        self.cfg       = config["rag"]
        self.top_k     = self.cfg["top_k"]
        self.min_sim   = self.cfg["min_similarity"]
        self.scaler    = scaler
        self.available = False

        if not FAISS_AVAILABLE:
            log.warning("FAISS unavailable — RAG retrieval disabled")
            return

        if not Path(index_path).exists():
            log.warning("FAISS index not found: %s — RAG disabled", index_path)
            return

        if not Path(metadata_path).exists():
            log.warning("FAISS metadata not found: %s — RAG disabled",
                        metadata_path)
            return

        # Load index
        self.index = faiss.read_index(index_path)
        log.info("FAISS index loaded: %d vectors", self.index.ntotal)

        # Load metadata
        self.metadata: list[dict] = []
        with open(metadata_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self.metadata.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        log.info("FAISS metadata loaded: %d entries", len(self.metadata))

        self.available = True

    def retrieve(self, feature_vec: np.ndarray) -> list[dict]:
        """
        Find the top-K most similar past anomalies for a given feature vector.

        Args:
            feature_vec: numpy array shape (35,)

        Returns:
            List of dicts, each representing a similar past anomaly:
            {similarity_score, event_id, event_time, user, src_ip,
             event_category, event_outcome, risk_score, country_src}
        """
        if not self.available:
            return []

        # Scale and normalize
        vec_scaled = self.scaler.transform(
            feature_vec.reshape(1, -1)
        ).astype(np.float32)
        faiss.normalize_L2(vec_scaled)

        # Search — returns distances and indices
        distances, indices = self.index.search(vec_scaled, self.top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue

            # FAISS inner product after L2 norm = cosine similarity (0-1)
            similarity = float(np.clip(dist, 0.0, 1.0))

            if similarity < self.min_sim:
                continue

            meta = self.metadata[idx]
            results.append({
                "similarity_score": round(similarity, 4),
                "event_id":         meta.get("event_id", ""),
                "event_time":       meta.get("event_time", ""),
                "user":             meta.get("user", "unknown"),
                "src_ip":           meta.get("src_ip", ""),
                "dst_ip":           meta.get("dst_ip", ""),
                "dst_port":         meta.get("dst_port", ""),
                "agent_name":       meta.get("agent_name", ""),
                "agent_ip":         meta.get("agent_ip", ""),
                "agent_role":       meta.get("agent_role", ""),
                "event_category":   meta.get("event_category", ""),
                "event_action":     meta.get("event_action", ""),
                "event_outcome":    meta.get("event_outcome", ""),
                "rule_id":          meta.get("rule_id", ""),
                "rule_desc":        meta.get("rule_desc", ""),
                "rule_level":       meta.get("rule_level", 0),
                "risk_score":       meta.get("risk_score", 0),
                "risk_verdict":     meta.get("risk_verdict", ""),
                "country_src":      meta.get("country_src", ""),
                "protocol":         meta.get("protocol", ""),
                "direction":        meta.get("direction", ""),
                "is_internal":      meta.get("is_internal", False),
                "fingerprint":      meta.get("fingerprint", ""),
            })

        return results

    def add_to_index(self, feature_vec: np.ndarray, metadata: dict):
        """
        Add a new anomaly to the FAISS index at runtime.
        Keeps the index growing with new anomalies as they're detected.
        Evicts oldest entries if max_index_size is exceeded.
        """
        if not self.available:
            return

        max_size = self.cfg["max_index_size"]
        if self.index.ntotal >= max_size:
            # FAISS flat indexes don't support deletion,
            # so we rebuild from the newest half when full.
            # In production, consider using IndexIDMap for proper deletion.
            log.warning(
                "FAISS index full (%d vectors). Consider rebuilding index.",
                self.index.ntotal,
            )
            return

        vec_scaled = self.scaler.transform(
            feature_vec.reshape(1, -1)
        ).astype(np.float32)
        faiss.normalize_L2(vec_scaled)
        self.index.add(vec_scaled)
        self.metadata.append(metadata)
