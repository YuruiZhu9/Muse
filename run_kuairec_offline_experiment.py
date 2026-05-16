import argparse
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import DEVICE, EMB_DIM, MAX_HISTORY_LEN, RANKING_CONFIG, RANKING_TOP_K, RECALL_FUSE_SIZE, RERANK_CONFIG
from src.data_loader_kuairec import KuaiRecDataLoader
from src.evaluation import KuaiRecEvaluator, summarize_prediction_overlap
from src.ranking import RankingLoss, StateEnhancedRankingModel
from src.recall import HybridRecall
from src.rerank import ReRanker
from src.two_tower import TwoTowerModel, train_two_tower, build_two_tower_index


DEFAULT_INTERACTION_FILE = Path("Datasets/KuaiRec 2.0/data/big_matrix.csv")
DEFAULT_USER_INFERENCE_FILE = Path("LLM_part/user_inferences_big_train.jsonl")
DEFAULT_OUTPUT_FILE = Path("LLM_part/kuairec_offline_experiment_results.json")


@dataclass
class ExperimentMode:
    name: str
    use_llm_recall: bool
    use_llm_rank: bool
    use_semantic_rerank: bool


EXPERIMENTS: Dict[str, ExperimentMode] = {
    "din_only": ExperimentMode(
        name="din_only",
        use_llm_recall=False,
        use_llm_rank=False,
        use_semantic_rerank=False,
    ),
    "llm_recall_only": ExperimentMode(
        name="llm_recall_only",
        use_llm_recall=True,
        use_llm_rank=False,
        use_semantic_rerank=False,
    ),
    "llm_full": ExperimentMode(
        name="llm_full",
        use_llm_recall=True,
        use_llm_rank=True,
        use_semantic_rerank=True,
    ),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_inference_user_ids(user_inference_file: Path) -> List[int]:
    user_ids: List[int] = []
    with user_inference_file.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            obj = json.loads(line)
            user_id = extract_row_user_id(obj)
            if user_id is None:
                continue
            user_ids.append(user_id)
    return sorted(set(user_ids))


def load_inference_rows_map(user_inference_file: Path) -> Dict[int, Dict]:
    rows: Dict[int, Dict] = {}
    with user_inference_file.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            obj = json.loads(line)
            user_id = extract_row_user_id(obj)
            if user_id is None:
                continue
            rows[int(user_id)] = obj
    return rows


def extract_row_user_id(obj: Dict) -> Optional[int]:
    candidates = []

    if isinstance(obj, dict):
        basic = obj.get("user_basic_information")
        if isinstance(basic, dict):
            candidates.append(basic.get("user_id"))
        candidates.append(obj.get("user_id"))

        sample_id = obj.get("sample_id")
        if isinstance(sample_id, str):
            match = re.search(r"u(\d+)", sample_id)
            if match:
                candidates.append(match.group(1))

    for candidate in candidates:
        try:
            return int(candidate)
        except (TypeError, ValueError):
            continue

    return None


def load_interactions_for_users(interaction_file: Path, user_ids: Sequence[int]) -> pd.DataFrame:
    selected_set = set(int(user_id) for user_id in user_ids)
    chunks: List[pd.DataFrame] = []

    for chunk in pd.read_csv(
        interaction_file,
        usecols=["user_id", "video_id", "play_duration", "video_duration", "time", "date", "timestamp", "watch_ratio"],
        chunksize=300_000,
    ):
        filtered = chunk[chunk["user_id"].isin(selected_set)]
        if not filtered.empty:
            chunks.append(filtered)

    if not chunks:
        raise RuntimeError("No interactions found for the requested users.")

    interactions = pd.concat(chunks, ignore_index=True)
    interactions = interactions.dropna(subset=["user_id", "video_id", "timestamp", "watch_ratio"])
    interactions["user_id"] = interactions["user_id"].astype(int)
    interactions["video_id"] = interactions["video_id"].astype(int)
    interactions["timestamp"] = interactions["timestamp"].astype(int)
    interactions["watch_ratio"] = pd.to_numeric(interactions["watch_ratio"], errors="coerce").fillna(0.0)
    interactions["date"] = pd.to_datetime(interactions["date"].astype(str), format="%Y%m%d", errors="coerce")
    interactions = interactions.dropna(subset=["date"]).copy()
    interactions = interactions.sort_values(["user_id", "timestamp", "video_id"]).reset_index(drop=True)
    return interactions


def build_holdout_split(
    interactions: pd.DataFrame,
    max_eval_users: int,
    positive_watch_ratio: float,
    holdout_positive_count: int,
    min_history_length: int,
    min_total_interactions: int,
) -> Tuple[List[int], Dict[int, int], Dict[int, List[int]]]:
    selected_users: List[int] = []
    max_timestamp_by_user: Dict[int, int] = {}
    ground_truth: Dict[int, List[int]] = {}

    for user_id, group in interactions.groupby("user_id", sort=True):
        group = group.sort_values(["timestamp", "video_id"]).reset_index(drop=True)
        if len(group) < min_total_interactions:
            continue

        positive_mask = group["watch_ratio"] >= positive_watch_ratio
        positive_indices = group.index[positive_mask].tolist()
        if len(positive_indices) < holdout_positive_count:
            continue

        holdout_indices = positive_indices[-holdout_positive_count:]
        split_index = holdout_indices[0]
        if split_index < min_history_length:
            continue

        history = group.iloc[:split_index]
        if len(history) < min_history_length:
            continue

        selected_users.append(int(user_id))
        max_timestamp_by_user[int(user_id)] = int(history.iloc[-1]["timestamp"])
        ground_truth[int(user_id)] = [int(item_id) for item_id in group.iloc[holdout_indices]["video_id"].tolist()]

        if len(selected_users) >= max_eval_users:
            break

    if not selected_users:
        raise RuntimeError("No users satisfied the holdout split constraints.")

    return selected_users, max_timestamp_by_user, ground_truth


def build_anchor_based_split(
    interactions: pd.DataFrame,
    inference_rows: Dict[int, Dict],
    max_eval_users: int,
    positive_watch_ratio: float,
    min_history_length: int,
    min_total_interactions: int,
    require_unseen_ground_truth: bool = True,
) -> Tuple[List[int], Dict[int, int], Dict[int, List[int]]]:
    selected_users: List[int] = []
    max_timestamp_by_user: Dict[int, int] = {}
    ground_truth: Dict[int, List[int]] = {}

    grouped_interactions = {
        int(user_id): group.sort_values(["timestamp", "video_id"]).reset_index(drop=True)
        for user_id, group in interactions.groupby("user_id", sort=True)
    }

    for user_id in sorted(inference_rows.keys()):
        row = inference_rows[user_id]
        anchor_date_raw = row.get("anchor_date")
        if not anchor_date_raw:
            continue

        group = grouped_interactions.get(int(user_id))
        if group is None or len(group) < min_total_interactions:
            continue

        anchor_date = pd.to_datetime(anchor_date_raw, errors="coerce")
        if pd.isna(anchor_date):
            continue
        anchor_date = anchor_date.normalize()

        history = group[group["date"] <= anchor_date].copy()
        future = group[(group["date"] > anchor_date) & (group["watch_ratio"] >= positive_watch_ratio)].copy()

        if len(history) < min_history_length or future.empty:
            continue

        if require_unseen_ground_truth:
            seen_items = set(int(item_id) for item_id in history["video_id"].tolist())
            future = future[~future["video_id"].isin(seen_items)].copy()
            if future.empty:
                continue

        selected_users.append(int(user_id))
        max_timestamp_by_user[int(user_id)] = int(history["timestamp"].max())
        deduped_future = list(dict.fromkeys(int(item_id) for item_id in future["video_id"].tolist()))
        ground_truth[int(user_id)] = deduped_future

        if len(selected_users) >= max_eval_users:
            break

    if not selected_users:
        raise RuntimeError("No users satisfied the anchor-date split constraints.")

    return selected_users, max_timestamp_by_user, ground_truth


@dataclass
class RankingSample:
    local_user_id: int
    history_item_ids: List[int]
    target_item_id: int
    ctr_label: float
    cvr_label: float


class RankingSampleDataset(Dataset):
    def __init__(self, samples: List[RankingSample], loader: KuaiRecDataLoader, use_llm_rank: bool, max_history_len: int):
        self.samples = samples
        self.loader = loader
        self.use_llm_rank = use_llm_rank
        self.max_history_len = max_history_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]
        user_features = self.loader.get_user_features(sample.local_user_id)
        item_features = self.loader.get_item_features(sample.target_item_id)

        result = {
            "user_id": sample.local_user_id,
            "history_item_ids": sample.history_item_ids[-self.max_history_len :],
            "target_item_id": sample.target_item_id,
            "user_features": user_features.feature_vector.astype(np.float32),
            "item_features": item_features.feature_vector.astype(np.float32),
            "ctr_label": np.float32(sample.ctr_label),
            "cvr_label": np.float32(sample.cvr_label),
        }

        if self.use_llm_rank:
            result["user_state_embs"] = self.loader.get_user_state_embs(sample.local_user_id).astype(np.float32)
            result["item_semantic_embs"] = self.loader.get_item_semantic_embs(sample.target_item_id).astype(np.float32)

        return result


def build_collate_fn(use_llm_rank: bool, max_history_len: int):
    def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)
        histories = np.zeros((batch_size, max_history_len), dtype=np.int64)

        for index, row in enumerate(batch):
            history = row["history_item_ids"][-max_history_len :]
            if history:
                offset_history = np.array([int(item_id) + 1 for item_id in history], dtype=np.int64)
                histories[index, -len(offset_history) :] = offset_history

        collated = {
            "user_id": torch.tensor([int(row["user_id"]) for row in batch], dtype=torch.long),
            "hist_item_ids": torch.tensor(histories, dtype=torch.long),
            "target_item_id": torch.tensor([int(row["target_item_id"]) + 1 for row in batch], dtype=torch.long),
            "user_features": torch.tensor(np.stack([row["user_features"] for row in batch]), dtype=torch.float32),
            "item_features": torch.tensor(np.stack([row["item_features"] for row in batch]), dtype=torch.float32),
            "ctr_label": torch.tensor([[float(row["ctr_label"])] for row in batch], dtype=torch.float32),
            "cvr_label": torch.tensor([[float(row["cvr_label"])] for row in batch], dtype=torch.float32),
        }

        if use_llm_rank:
            collated["user_state_embs"] = torch.tensor(
                np.stack([row["user_state_embs"] for row in batch]),
                dtype=torch.float32,
            )
            collated["item_semantic_embs"] = torch.tensor(
                np.stack([row["item_semantic_embs"] for row in batch]),
                dtype=torch.float32,
            )

        return collated

    return collate


def build_training_samples(
    loader: KuaiRecDataLoader,
    interactions: pd.DataFrame,
    positive_watch_ratio: float,
    negative_watch_ratio: float,
    negatives_per_positive: int,
    min_history_length: int,
    max_history_len: int,
    repeat_watch_ratio: float,
    seed: int,
    regression: bool = False,
) -> List[RankingSample]:
    rng = random.Random(seed)
    all_original_item_ids = sorted(loader.item_id_map.keys())
    samples: List[RankingSample] = []

    for original_user_id, group in interactions.groupby("user_id", sort=True):
        if original_user_id not in loader.user_id_map:
            continue

        group = group.sort_values(["timestamp", "video_id"]).reset_index(drop=True)
        local_user_id = loader.user_id_map[int(original_user_id)]
        interacted_original_items = set(int(item_id) for item_id in group["video_id"].tolist())
        negative_pool = [item_id for item_id in all_original_item_ids if item_id not in interacted_original_items]
        history_item_ids: List[int] = []

        for _, row in group.iterrows():
            original_item_id = int(row["video_id"])
            local_item_id = loader.item_id_map[original_item_id]
            watch_ratio = float(row["watch_ratio"])

            if len(history_item_ids) >= min_history_length:
                history_snapshot = history_item_ids[-max_history_len:]

                if watch_ratio >= positive_watch_ratio:
                    if regression:
                        # CTR: watch_ratio capped at 1.0 as watch-completion probability
                        # CVR: binary re-watch signal (1.0 if watch_ratio > 1.0)
                        ctr_label = min(watch_ratio, 1.0)
                        cvr_label = 1.0 if watch_ratio > 1.0 else 0.0
                    else:
                        ctr_label = 1.0
                        cvr_label = 1.0 if watch_ratio >= repeat_watch_ratio else 0.0
                    samples.append(
                        RankingSample(
                            local_user_id=local_user_id,
                            history_item_ids=history_snapshot.copy(),
                            target_item_id=local_item_id,
                            ctr_label=ctr_label,
                            cvr_label=cvr_label,
                        )
                    )

                    if negative_pool and negatives_per_positive > 0:
                        sampled_negatives = rng.sample(negative_pool, k=min(negatives_per_positive, len(negative_pool)))
                        for negative_original_item_id in sampled_negatives:
                            negative_local_item_id = loader.item_id_map[negative_original_item_id]
                            samples.append(
                                RankingSample(
                                    local_user_id=local_user_id,
                                    history_item_ids=history_snapshot.copy(),
                                    target_item_id=negative_local_item_id,
                                    ctr_label=0.0,
                                    cvr_label=0.0,
                                )
                            )
                elif watch_ratio <= negative_watch_ratio:
                    ctr_label = watch_ratio if regression else 0.0
                    samples.append(
                        RankingSample(
                            local_user_id=local_user_id,
                            history_item_ids=history_snapshot.copy(),
                            target_item_id=local_item_id,
                            ctr_label=ctr_label,
                            cvr_label=0.0,
                        )
                    )

            history_item_ids.append(local_item_id)

    return samples


def create_ranking_model(loader: KuaiRecDataLoader,
                         llm_proj_dim=None, semantic_num_heads=None, semantic_dropout=None) -> StateEnhancedRankingModel:
    proj_dim = llm_proj_dim if llm_proj_dim is not None else RANKING_CONFIG["llm_proj_dim"]
    n_heads = semantic_num_heads if semantic_num_heads is not None else RANKING_CONFIG.get("semantic_num_heads", 4)
    s_drop = semantic_dropout if semantic_dropout is not None else RANKING_CONFIG.get("semantic_dropout", 0.0)
    return StateEnhancedRankingModel(
        num_users=loader.num_users,
        num_items=loader.num_items + 1,
        user_feature_dim=32,
        item_feature_dim=32,
        embedding_dim=RANKING_CONFIG["user_id_dim"],
        history_len=MAX_HISTORY_LEN,
        llm_semantic_dim=EMB_DIM,
        llm_proj_dim=proj_dim,
        attention_hidden_dim=RANKING_CONFIG["attention_hidden_dim"],
        num_experts=RANKING_CONFIG["num_experts"],
        expert_hidden_dim=RANKING_CONFIG["expert_hidden_dim"],
        enable_llm_features=True,
        semantic_num_heads=n_heads,
        semantic_dropout=s_drop,
    ).to(DEVICE)


def train_ranking_model(
    loader: KuaiRecDataLoader,
    training_samples: List[RankingSample],
    use_llm_rank: bool,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    cvr_weight: float,
    early_stop_patience: int = 5,
    regression: bool = False,
    llm_proj_dim=None, semantic_num_heads=None, semantic_dropout=None,
) -> Tuple[StateEnhancedRankingModel, Dict[str, float]]:
    if not training_samples:
        raise RuntimeError("Training sample set is empty.")

    dataset = RankingSampleDataset(
        samples=training_samples,
        loader=loader,
        use_llm_rank=use_llm_rank,
        max_history_len=MAX_HISTORY_LEN,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=build_collate_fn(use_llm_rank=use_llm_rank, max_history_len=MAX_HISTORY_LEN),
    )

    model = create_ranking_model(loader, llm_proj_dim, semantic_num_heads, semantic_dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = RankingLoss(ctr_weight=1.0, cvr_weight=cvr_weight, regression=regression)
    epoch_losses: List[float] = []
    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch_index in range(epochs):
        model.train()
        batch_losses: List[float] = []

        for batch in dataloader:
            user_state_embs = batch.get("user_state_embs")
            item_semantic_embs = batch.get("item_semantic_embs")

            optimizer.zero_grad()
            outputs = model(
                user_id=batch["user_id"].to(DEVICE),
                hist_item_ids=batch["hist_item_ids"].to(DEVICE),
                target_item_id=batch["target_item_id"].to(DEVICE),
                user_features=batch["user_features"].to(DEVICE),
                item_features=batch["item_features"].to(DEVICE),
                user_state_embs=user_state_embs.to(DEVICE) if user_state_embs is not None else None,
                item_semantic_embs=item_semantic_embs.to(DEVICE) if item_semantic_embs is not None else None,
                enable_llm_features=use_llm_rank,
            )
            loss, _ = loss_fn(
                outputs["ctr"],
                outputs["cvr"],
                batch["ctr_label"].to(DEVICE),
                batch["cvr_label"].to(DEVICE),
            )
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        mean_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        epoch_losses.append(mean_loss)
        print(f"[train] mode={'llm' if use_llm_rank else 'base'} epoch={epoch_index + 1}/{epochs} loss={mean_loss:.6f}")

        if mean_loss < best_loss - 1e-4:
            best_loss = mean_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"[train] early stop at epoch {epoch_index + 1} (no improvement for {early_stop_patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "train_samples": float(len(training_samples)),
        "epochs": float(epoch_index + 1),
        "final_loss": float(epoch_losses[-1]) if epoch_losses else 0.0,
        "best_loss": float(min(epoch_losses)) if epoch_losses else 0.0,
    }


def score_candidates_for_user(
    loader: KuaiRecDataLoader,
    ranking_model: StateEnhancedRankingModel,
    recall_module: HybridRecall,
    local_user_id: int,
    use_llm_recall: bool,
    use_llm_rank: bool,
    use_semantic_rerank: bool,
    top_k: int,
) -> Dict[str, List[int]]:
    original_flag = recall_module.enable_llm_features
    recall_module.enable_llm_features = use_llm_recall
    try:
        recall_results = recall_module.recall(local_user_id)
    finally:
        recall_module.enable_llm_features = original_flag

    fused_candidates = recall_module.fuse_recall_results(recall_results, final_size=RECALL_FUSE_SIZE)
    if not fused_candidates:
        return {
            "fused_candidates": [],
            "ranking_items": [],
            "final_items": [],
        }

    user_features = loader.get_user_features(local_user_id)
    user_history = loader.get_user_history(local_user_id, MAX_HISTORY_LEN)
    hist_padded = [0] * MAX_HISTORY_LEN
    if user_history:
        offset_history = [int(item_id) + 1 for item_id in user_history[-MAX_HISTORY_LEN:]]
        hist_padded[-len(offset_history) :] = offset_history

    batch_size = len(fused_candidates)
    user_ids = torch.tensor([local_user_id] * batch_size, dtype=torch.long, device=DEVICE)
    hist_item_ids = torch.tensor([hist_padded] * batch_size, dtype=torch.long, device=DEVICE)
    target_item_ids = torch.tensor([int(item_id) + 1 for item_id in fused_candidates], dtype=torch.long, device=DEVICE)
    user_feat_tensor = torch.tensor(
        [user_features.feature_vector.tolist()] * batch_size,
        dtype=torch.float32,
        device=DEVICE,
    )
    item_feat_tensor = torch.tensor(
        [loader.get_item_features(item_id).feature_vector.tolist() for item_id in fused_candidates],
        dtype=torch.float32,
        device=DEVICE,
    )

    user_state_embs = None
    item_semantic_embs = None
    if use_llm_rank:
        user_state = loader.get_user_state_embs(local_user_id)
        user_state_embs = torch.tensor(user_state, dtype=torch.float32, device=DEVICE).unsqueeze(0).expand(batch_size, -1, -1)
        item_semantic_embs = torch.tensor(
            loader.get_batch_item_semantic_embs(fused_candidates),
            dtype=torch.float32,
            device=DEVICE,
        )

    ranking_model.eval()
    with torch.no_grad():
        outputs = ranking_model(
            user_id=user_ids,
            hist_item_ids=hist_item_ids,
            target_item_id=target_item_ids,
            user_features=user_feat_tensor,
            item_features=item_feat_tensor,
            user_state_embs=user_state_embs,
            item_semantic_embs=item_semantic_embs,
            enable_llm_features=use_llm_rank,
        )

    ctr_scores = outputs["ctr"].squeeze(-1).detach().cpu().numpy()
    cvr_scores = outputs["cvr"].squeeze(-1).detach().cpu().numpy()
    ranked_items = []
    ctr_cvr_records = []
    for idx, (item_id, score) in enumerate(zip(fused_candidates, ctr_scores)):
        item_features = loader.get_item_features(item_id)
        ranked_items.append(
            {
                "item_id": item_id,
                "original_item_id": loader.original_item_id(item_id),
                "score": float(score),
                "category_id": item_features.category_id,
            }
        )
        ctr_cvr_records.append({
            "original_item_id": int(loader.original_item_id(item_id)),
            "ctr_pred": float(score),
            "cvr_pred": float(cvr_scores[idx]),
        })

    ranked_items.sort(key=lambda row: row["score"], reverse=True)
    ranked_items = ranked_items[:RANKING_TOP_K]
    ranking_original_ids = [int(row["original_item_id"]) for row in ranked_items]

    if use_semantic_rerank and ranked_items:
        reranker = ReRanker(loader, strategy="dpp")
        rerank_scores = np.array([row["score"] for row in ranked_items], dtype=np.float32)
        rerank_semantic = loader.get_batch_item_semantic_embs([row["item_id"] for row in ranked_items])
        ranked_items = reranker.rerank(
            ranked_items,
            scores=rerank_scores,
            semantic_embs=rerank_semantic,
            final_size=min(top_k, len(ranked_items)),
            seen_item_ids=set(user_history),
            filter_seen_items=RERANK_CONFIG.get("filter_seen_items", True),
            prefix_diversity_top_n=RERANK_CONFIG.get("prefix_diversity_top_n", 5),
            max_prefix_same_category=RERANK_CONFIG.get("max_prefix_same_category", 2),
            max_consecutive_same_category=RERANK_CONFIG.get("max_consecutive_same_category", 1),
            max_adjacent_semantic_similarity=RERANK_CONFIG.get("max_adjacent_semantic_similarity", 0.92),
            lambda_diversity=0.5,
        )
    else:
        ranked_items = ranked_items[:top_k]

    return {
        "fused_candidates": [int(loader.original_item_id(item_id)) for item_id in fused_candidates],
        "ranking_items": ranking_original_ids,
        "final_items": [int(row["original_item_id"]) for row in ranked_items[:top_k]],
        "ctr_cvr_records": ctr_cvr_records,
    }


def compute_ctr_cvr_auc(
    ctr_cvr_records: Dict[int, List[Dict]],
    ground_truth: Dict[int, List[int]],
) -> Dict[str, float]:
    """Compute CTR/CVR AUC and GAUC (Group AUC, per-user weighted average)."""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return {"ctr_auc": float("nan"), "cvr_auc": float("nan"),
                "ctr_gauc": float("nan"), "cvr_gauc": float("nan")}

    all_ctr_preds: List[float] = []
    all_ctr_labels: List[float] = []
    all_cvr_preds: List[float] = []
    all_cvr_labels: List[float] = []

    # Per-user AUC for GAUC computation
    user_ctr_aucs: List[float] = []
    user_cvr_aucs: List[float] = []
    user_ctr_weights: List[int] = []
    user_cvr_weights: List[int] = []

    for user_id, records in ctr_cvr_records.items():
        true_set = set(int(item_id) for item_id in ground_truth.get(user_id, []))
        user_ctr_p: List[float] = []
        user_ctr_l: List[float] = []
        user_cvr_p: List[float] = []
        user_cvr_l: List[float] = []
        n_pos = 0

        for rec in records:
            label = 1.0 if rec["original_item_id"] in true_set else 0.0
            if label > 0:
                n_pos += 1
            all_ctr_preds.append(rec["ctr_pred"])
            all_ctr_labels.append(label)
            all_cvr_preds.append(rec["cvr_pred"])
            all_cvr_labels.append(label)
            user_ctr_p.append(rec["ctr_pred"])
            user_ctr_l.append(label)
            user_cvr_p.append(rec["cvr_pred"])
            user_cvr_l.append(label)

        if len(set(user_ctr_l)) >= 2:
            user_ctr_aucs.append(float(roc_auc_score(user_ctr_l, user_ctr_p)))
            user_ctr_weights.append(max(n_pos, 1))
        if len(set(user_cvr_l)) >= 2:
            user_cvr_aucs.append(float(roc_auc_score(user_cvr_l, user_cvr_p)))
            user_cvr_weights.append(max(n_pos, 1))

    result: Dict[str, float] = {}
    if len(set(all_ctr_labels)) >= 2:
        result["ctr_auc"] = float(roc_auc_score(all_ctr_labels, all_ctr_preds))
        result["cvr_auc"] = float(roc_auc_score(all_cvr_labels, all_cvr_preds))
    else:
        result["ctr_auc"] = float("nan")
        result["cvr_auc"] = float("nan")

    result["ctr_gauc"] = (
        float(np.average(user_ctr_aucs, weights=user_ctr_weights))
        if user_ctr_aucs else float("nan")
    )
    result["cvr_gauc"] = (
        float(np.average(user_cvr_aucs, weights=user_cvr_weights))
        if user_cvr_aucs else float("nan")
    )
    return result


def compute_stage_hit_rate(
    ground_truth: Dict[int, List[int]],
    predictions: Dict[int, List[int]],
) -> float:
    hits: List[float] = []
    for user_id, true_items in ground_truth.items():
        true_set = set(int(item_id) for item_id in true_items)
        pred_set = set(int(item_id) for item_id in predictions.get(user_id, []))
        hits.append(1.0 if pred_set & true_set else 0.0)
    return float(np.mean(hits)) if hits else 0.0


def evaluate_experiment(
    loader: KuaiRecDataLoader,
    ranking_model: StateEnhancedRankingModel,
    experiment: ExperimentMode,
    ground_truth: Dict[int, List[int]],
    top_k: int,
    two_tower_item_embs=None,
    two_tower_model=None,
) -> Dict:
    recall_module = HybridRecall(data_loader=loader, enable_llm_features=experiment.use_llm_recall)
    if two_tower_item_embs is not None:
        recall_module.set_two_tower_index(two_tower_item_embs, two_tower_model)
    predictions: Dict[int, List[int]] = {}
    candidate_predictions: Dict[int, List[int]] = {}
    ranking_predictions: Dict[int, List[int]] = {}
    ctr_cvr_records: Dict[int, List[Dict]] = {}

    for local_user_id in range(loader.num_users):
        original_user_id = loader.original_user_id(local_user_id)
        stage_outputs = score_candidates_for_user(
            loader=loader,
            ranking_model=ranking_model,
            recall_module=recall_module,
            local_user_id=local_user_id,
            use_llm_recall=experiment.use_llm_recall,
            use_llm_rank=experiment.use_llm_rank,
            use_semantic_rerank=experiment.use_semantic_rerank,
            top_k=top_k,
        )
        candidate_predictions[original_user_id] = stage_outputs["fused_candidates"]
        ranking_predictions[original_user_id] = stage_outputs["ranking_items"]
        predictions[original_user_id] = stage_outputs["final_items"]
        ctr_cvr_records[original_user_id] = stage_outputs.get("ctr_cvr_records", [])

    evaluator = KuaiRecEvaluator(ground_truth=ground_truth, catalog_size=loader.num_items)
    metrics = evaluator.evaluate(predictions, k=top_k)
    metrics["candidate_hit_rate"] = compute_stage_hit_rate(ground_truth, candidate_predictions)
    metrics["ranking_hit_rate"] = compute_stage_hit_rate(ground_truth, ranking_predictions)
    ctr_cvr_auc = compute_ctr_cvr_auc(ctr_cvr_records, ground_truth)
    metrics.update(ctr_cvr_auc)
    return {
        "metrics": metrics,
        "predictions": predictions,
        "candidate_predictions": candidate_predictions,
        "ranking_predictions": ranking_predictions,
        "ctr_cvr_records": {str(k): v for k, v in ctr_cvr_records.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline KuaiRec recommendation experiments.")
    parser.add_argument("--interaction-file", type=Path, default=DEFAULT_INTERACTION_FILE)
    parser.add_argument("--user-inference-file", type=Path, default=DEFAULT_USER_INFERENCE_FILE)
    parser.add_argument("--experiment", choices=["din_only", "llm_recall_only", "llm_full", "compare_all"], default="compare_all")
    parser.add_argument("--text-encoder", choices=["hash", "qwen"], default="hash")
    parser.add_argument("--cutoff-mode", choices=["auto", "anchor_date", "last_positive"], default="auto")
    parser.add_argument("--max-eval-users", type=int, default=20)
    parser.add_argument("--holdout-positive-count", type=int, default=1)
    parser.add_argument("--positive-watch-ratio", type=float, default=0.8)
    parser.add_argument("--negative-watch-ratio", type=float, default=0.3)
    parser.add_argument("--repeat-watch-ratio", type=float, default=1.0)
    parser.add_argument("--min-history-length", type=int, default=5)
    parser.add_argument("--min-total-interactions", type=int, default=10)
    parser.add_argument("--allow-seen-ground-truth", action="store_true")
    parser.add_argument("--negatives-per-positive", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--regression", action="store_true",
                        help="Use continuous watch_ratio as CTR/CVR labels (MSE loss)")
    parser.add_argument("--llm-proj-dim", type=int, default=None,
                        help="Override llm_proj_dim from config")
    parser.add_argument("--semantic-num-heads", type=int, default=None,
                        help="Override semantic_num_heads from config")
    parser.add_argument("--semantic-dropout", type=float, default=None,
                        help="Override semantic_dropout from config")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--cvr-weight", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    start_time = time.time()
    inference_rows = load_inference_rows_map(args.user_inference_file)
    inference_user_ids = sorted(inference_rows.keys())
    raw_interactions = load_interactions_for_users(args.interaction_file, inference_user_ids)

    has_anchor_dates = any(row.get("anchor_date") for row in inference_rows.values())
    cutoff_mode = args.cutoff_mode
    if cutoff_mode == "auto":
        cutoff_mode = "anchor_date" if has_anchor_dates else "last_positive"

    if cutoff_mode == "anchor_date":
        selected_user_ids, max_timestamp_by_user, ground_truth = build_anchor_based_split(
            interactions=raw_interactions,
            inference_rows=inference_rows,
            max_eval_users=args.max_eval_users,
            positive_watch_ratio=args.positive_watch_ratio,
            min_history_length=args.min_history_length,
            min_total_interactions=args.min_total_interactions,
            require_unseen_ground_truth=not args.allow_seen_ground_truth,
        )
    else:
        selected_user_ids, max_timestamp_by_user, ground_truth = build_holdout_split(
            interactions=raw_interactions,
            max_eval_users=args.max_eval_users,
            positive_watch_ratio=args.positive_watch_ratio,
            holdout_positive_count=args.holdout_positive_count,
            min_history_length=args.min_history_length,
            min_total_interactions=args.min_total_interactions,
        )

    print(f"[split] selected_users={len(selected_user_ids)} text_encoder={args.text_encoder} cutoff_mode={cutoff_mode}")

    loader = KuaiRecDataLoader(
        interaction_file=args.interaction_file,
        user_inference_file=args.user_inference_file,
        user_limit=None,
        selected_user_ids=selected_user_ids,
        max_timestamp_by_user=max_timestamp_by_user,
        emb_dim=EMB_DIM,
        text_encoder_backend=args.text_encoder,
    )
    print(f"[loader] {loader}")

    training_samples = build_training_samples(
        loader=loader,
        interactions=loader._interactions.copy(),
        positive_watch_ratio=args.positive_watch_ratio,
        negative_watch_ratio=args.negative_watch_ratio,
        negatives_per_positive=args.negatives_per_positive,
        min_history_length=args.min_history_length,
        max_history_len=MAX_HISTORY_LEN,
        repeat_watch_ratio=args.repeat_watch_ratio,
        seed=args.seed,
        regression=args.regression,
    )
    print(f"[train-data] samples={len(training_samples)}")

    # --- Train Two-Tower recall model ---
    two_tower_item_embs: Optional[np.ndarray] = None
    tt_model: Optional[TwoTowerModel] = None
    print("[two-tower] training recall model...")
    user_state_embs_full = loader.load_user_state_embs()
    item_semantic_embs_full = loader.load_item_semantic_embs()
    if user_state_embs_full is not None and item_semantic_embs_full is not None:
        # Only use POSITIVE pairs for two-tower (ctr_label > 0)
        positive_pairs = [
            (sample.local_user_id, sample.target_item_id)
            for sample in training_samples
            if sample.ctr_label > 0 and sample.local_user_id < len(user_state_embs_full)
        ]
        # Also add all historical user-item interactions as positive pairs (richer signal)
        for original_user_id, group in raw_interactions.groupby("user_id", sort=True):
            if original_user_id not in loader.user_id_map:
                continue
            local_user_id = loader.user_id_map[int(original_user_id)]
            if local_user_id >= len(user_state_embs_full):
                continue
            for _, row in group.iterrows():
                original_item_id = int(row["video_id"])
                if original_item_id in loader.item_id_map:
                    positive_pairs.append((local_user_id, loader.item_id_map[original_item_id]))
        # Deduplicate
        positive_pairs = list(set(positive_pairs))
        print(f"[two-tower] positive pairs: {len(positive_pairs)} (training only: {sum(1 for s in training_samples if s.ctr_label > 0)})")
        if len(positive_pairs) >= 10:
            tt_model = TwoTowerModel(
                state_dim=user_state_embs_full.shape[2] if len(user_state_embs_full.shape) > 2 else 2560,
                semantic_dim=item_semantic_embs_full.shape[1] if len(item_semantic_embs_full.shape) > 1 else 2560,
            )
            tt_model = train_two_tower(
                tt_model, user_state_embs_full, item_semantic_embs_full,
                positive_pairs, epochs=20, batch_size=512, lr=1e-3,
            )
            two_tower_item_embs = build_two_tower_index(tt_model, item_semantic_embs_full)
            print(f"[two-tower] built item index: {two_tower_item_embs.shape}")
    else:
        print("[two-tower] skipping — missing user state or item semantic embeddings")

    base_model: Optional[StateEnhancedRankingModel] = None
    llm_model: Optional[StateEnhancedRankingModel] = None
    model_summaries: Dict[str, Dict[str, float]] = {}

    requested_experiments = (
        ["din_only", "llm_recall_only", "llm_full"]
        if args.experiment == "compare_all"
        else [args.experiment]
    )

    if any(name in requested_experiments for name in ["din_only", "llm_recall_only"]):
        base_model, model_summaries["base"] = train_ranking_model(
            loader=loader,
            training_samples=training_samples,
            use_llm_rank=False,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            cvr_weight=args.cvr_weight,
            early_stop_patience=args.early_stop_patience,
            regression=args.regression,
            llm_proj_dim=args.llm_proj_dim, semantic_num_heads=args.semantic_num_heads,
            semantic_dropout=args.semantic_dropout,
        )

    if "llm_full" in requested_experiments:
        llm_model, model_summaries["llm"] = train_ranking_model(
            loader=loader,
            training_samples=training_samples,
            use_llm_rank=True,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            cvr_weight=args.cvr_weight,
            early_stop_patience=args.early_stop_patience,
            regression=args.regression,
            llm_proj_dim=args.llm_proj_dim, semantic_num_heads=args.semantic_num_heads,
            semantic_dropout=args.semantic_dropout,
        )

    experiment_results: Dict[str, Dict] = {}

    for experiment_name in requested_experiments:
        experiment = EXPERIMENTS[experiment_name]
        ranking_model = llm_model if experiment.use_llm_rank else base_model
        if ranking_model is None:
            raise RuntimeError(f"No trained model available for experiment {experiment_name}.")

        print(
            f"[eval] experiment={experiment_name} "
            f"llm_recall={experiment.use_llm_recall} "
            f"llm_rank={experiment.use_llm_rank} "
            f"semantic_rerank={experiment.use_semantic_rerank}"
        )
        experiment_results[experiment_name] = evaluate_experiment(
            loader=loader,
            ranking_model=ranking_model,
            experiment=experiment,
            ground_truth=ground_truth,
            top_k=args.top_k,
            two_tower_item_embs=two_tower_item_embs,
            two_tower_model=tt_model,
        )
        print(f"[metrics] {experiment_name} {experiment_results[experiment_name]['metrics']}")

    overlap_summary: Dict[str, Dict[str, float]] = {}
    if "din_only" in experiment_results and "llm_full" in experiment_results:
        overlap_summary["din_vs_llm_full"] = summarize_prediction_overlap(
            experiment_results["din_only"]["predictions"],
            experiment_results["llm_full"]["predictions"],
        )
    if "din_only" in experiment_results and "llm_recall_only" in experiment_results:
        overlap_summary["din_vs_llm_recall_only"] = summarize_prediction_overlap(
            experiment_results["din_only"]["predictions"],
            experiment_results["llm_recall_only"]["predictions"],
        )

    serializable_results = {
        "config": {
            "interaction_file": str(args.interaction_file),
            "user_inference_file": str(args.user_inference_file),
            "experiment": args.experiment,
            "text_encoder": args.text_encoder,
            "cutoff_mode": cutoff_mode,
            "max_eval_users": args.max_eval_users,
            "holdout_positive_count": args.holdout_positive_count,
            "positive_watch_ratio": args.positive_watch_ratio,
            "negative_watch_ratio": args.negative_watch_ratio,
            "repeat_watch_ratio": args.repeat_watch_ratio,
            "min_history_length": args.min_history_length,
            "min_total_interactions": args.min_total_interactions,
            "allow_seen_ground_truth": args.allow_seen_ground_truth,
            "negatives_per_positive": args.negatives_per_positive,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "cvr_weight": args.cvr_weight,
            "top_k": args.top_k,
            "seed": args.seed,
        },
        "selected_user_ids": selected_user_ids,
        "ground_truth": ground_truth,
        "model_summaries": model_summaries,
        "experiments": experiment_results,
        "overlap_summary": overlap_summary,
        "runtime_seconds": round(time.time() - start_time, 3),
    }

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(serializable_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] saved results to {args.output_file}")


if __name__ == "__main__":
    main()
