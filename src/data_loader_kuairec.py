"""
KuaiRec-backed DataLoader for MuseRecSys smoke/integration tests.

This loader replaces the mock data path with:
1. Real KuaiRec user interactions
2. Real user/item metadata
3. LLM-generated user state text from JSONL
4. Deterministic 2560-d semantic embeddings derived from text

The semantic encoder is intentionally lightweight and deterministic so the
end-to-end recommendation pipeline can be exercised on a small test slice.
"""

from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

from LLM_part.student_schema import COMPACT_FIELD_NAMES, compact_state_texts_from_any

from .data_loader import UserFeatures, ItemFeatures, UserHistory


DEFAULT_INTERACTION_FILE = Path("Datasets/KuaiRec 2.0/data/big_matrix.csv")
DEFAULT_USER_FEATURES_FILE = Path("Datasets/KuaiRec 2.0/data/user_features.csv")
DEFAULT_USER_FEATURES_RAW_FILE = Path("Datasets/user_features_raw.csv")
DEFAULT_ITEM_TEXT_FILE = Path("Datasets/KuaiRec 2.0/data/kuairec_caption_category.csv")
DEFAULT_ITEM_RAW_CATEGORY_FILE = Path("Datasets/video_raw_categories_multi.csv")
DEFAULT_ITEM_DAILY_FILE = Path("Datasets/KuaiRec 2.0/data/item_daily_features.csv")
DEFAULT_USER_INFERENCE_FILE = Path("LLM_part/user_inferences_big_50_real.jsonl")
DEFAULT_TEXT_ENCODER_BACKEND = os.getenv("KUAIREC_TEXT_ENCODER", "hash").lower()
DEFAULT_EMBEDDING_MODEL_ID = os.getenv("KUAIREC_EMBEDDING_MODEL_ID", "Qwen/Qwen3-Embedding-4B")

USER_STATE_FIELDS = [
    "long_term_intent",
    "life_stage",
    "psychological_demand",
    "retrieval_suggestions",
    "interest_growth_points",
]


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text in {"", "UNKNOWN", "[]", "nan", "None"}:
        return ""
    return text


def _age_range_to_midpoint(age_range: str) -> int:
    age_range = _clean_text(age_range)
    if not age_range:
        return 28
    if "-" in age_range:
        left, right = age_range.split("-", 1)
        try:
            return int((int(left) + int(right)) / 2)
        except ValueError:
            return 28
    return 28


def _active_degree_to_float(active_degree: str) -> float:
    mapping = {
        "low_active": 0.25,
        "middle_active": 0.5,
        "high_active": 0.75,
        "full_active": 1.0,
    }
    return mapping.get(_clean_text(active_degree), 0.5)


def _gender_to_int(gender: str) -> int:
    gender = _clean_text(gender).upper()
    if gender in {"F", "FEMALE"}:
        return 1
    return 0


def _parse_topic_tags(topic_tag: str) -> List[str]:
    topic_tag = _clean_text(topic_tag)
    if not topic_tag:
        return []
    try:
        value = ast.literal_eval(topic_tag)
        if isinstance(value, list):
            return [str(x) for x in value[:5]]
    except Exception:
        pass
    return [x.strip() for x in topic_tag.split(",") if x.strip()][:5]


def _extract_row_user_id(obj: Dict) -> Optional[int]:
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


class HashSemanticEncoder:
    """Deterministic char-ngram hashing encoder with fixed 2560-d output."""

    def __init__(self, dim: int = 2560):
        self.dim = dim
        self.vectorizer = HashingVectorizer(
            n_features=dim,
            analyzer="char",
            ngram_range=(2, 4),
            alternate_sign=False,
            norm=None,
            lowercase=False,
        )

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        sparse = self.vectorizer.transform(texts)
        dense = sparse.astype(np.float32).toarray()
        dense = normalize(dense, norm="l2", axis=1)
        return dense.astype(np.float32)


class QwenSemanticEncoder:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL_ID, batch_size: int = 8, max_length: int = 512):
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.F = F
        self.batch_size = batch_size
        self.max_length = max_length

        hf_kwargs = {"trust_remote_code": True}
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            hf_kwargs["token"] = hf_token

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **hf_kwargs)
        self.model = AutoModel.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            **hf_kwargs,
        )
        self.model.eval()

    @staticmethod
    def _mean_pooling(model_output, attention_mask, torch_module):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch_module.sum(token_embeddings * input_mask_expanded, 1) / torch_module.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 2560), dtype=np.float32)

        outputs: List[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [text if text and text.strip() else "空文本" for text in texts[start : start + self.batch_size]]
            encoded_input = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.model.device)

            with self.torch.no_grad():
                model_output = self.model(**encoded_input)

            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"], self.torch)
            sentence_embeddings = self.F.normalize(sentence_embeddings, p=2, dim=1)
            outputs.append(sentence_embeddings.detach().cpu().numpy().astype(np.float32))

        return np.concatenate(outputs, axis=0)


class KuaiRecDataLoader:
    """Real-data loader for small-slice KuaiRec pipeline validation."""

    def __init__(
        self,
        interaction_file: Path | str = DEFAULT_INTERACTION_FILE,
        user_inference_file: Path | str = DEFAULT_USER_INFERENCE_FILE,
        user_limit: Optional[int] = 50,
        selected_user_ids: Optional[List[int]] = None,
        max_timestamp_by_user: Optional[Dict[int, int]] = None,
        emb_dim: int = 2560,
        cooccurrence_history_limit: int = 200,
        text_encoder_backend: Optional[str] = None,
    ):
        self.interaction_file = Path(interaction_file)
        self.user_inference_file = Path(user_inference_file)
        self.user_limit = user_limit
        self.selected_user_ids = set(int(user_id) for user_id in selected_user_ids) if selected_user_ids else None
        self.max_timestamp_by_user = {int(user_id): int(ts) for user_id, ts in (max_timestamp_by_user or {}).items()}
        self.emb_dim = emb_dim
        self.cooccurrence_history_limit = cooccurrence_history_limit
        self.text_encoder_backend = (text_encoder_backend or DEFAULT_TEXT_ENCODER_BACKEND).lower()
        if self.text_encoder_backend == "qwen":
            self.text_encoder = QwenSemanticEncoder(
                model_name=DEFAULT_EMBEDDING_MODEL_ID,
                batch_size=int(os.getenv("KUAIREC_EMBED_BATCH_SIZE", "8")),
                max_length=int(os.getenv("KUAIREC_EMBED_MAX_LENGTH", "512")),
            )
        else:
            self.text_encoder = HashSemanticEncoder(dim=emb_dim)

        self.selected_original_user_ids: List[int] = []
        self.user_id_map: Dict[int, int] = {}
        self.inverse_user_id_map: Dict[int, int] = {}
        self.item_id_map: Dict[int, int] = {}
        self.inverse_item_id_map: Dict[int, int] = {}

        self.num_users = 0
        self.num_items = 0

        self._user_features: Dict[int, UserFeatures] = {}
        self._item_features: Dict[int, ItemFeatures] = {}
        self._user_history: Dict[int, UserHistory] = {}
        self._user_state_embs: Optional[np.ndarray] = None
        self._item_semantic_embs: Optional[np.ndarray] = None
        self._item_texts_by_local_id: Dict[int, str] = {}
        self._item_category_names_by_local_id: Dict[int, str] = {}
        self._hot_items_cache: Optional[List[int]] = None
        self._cooccurrence_cache: Optional[Dict[Tuple[int, int], int]] = None
        self._all_interactions: Optional[pd.DataFrame] = None
        self._interactions: Optional[pd.DataFrame] = None

        self._build()

    # ------------------------------------------------------------------
    # Build pipeline assets
    # ------------------------------------------------------------------

    def _build(self):
        user_inference_rows = self._load_user_inferences()
        loaded_user_ids = sorted(user_inference_rows.keys())
        if self.selected_user_ids is not None:
            self.selected_original_user_ids = [user_id for user_id in loaded_user_ids if user_id in self.selected_user_ids]
        else:
            self.selected_original_user_ids = loaded_user_ids
        if self.user_limit is not None:
            self.selected_original_user_ids = self.selected_original_user_ids[: self.user_limit]
        if not self.selected_original_user_ids:
            raise RuntimeError("No user inference rows matched the requested users.")

        self.user_id_map = {uid: idx for idx, uid in enumerate(self.selected_original_user_ids)}
        self.inverse_user_id_map = {idx: uid for uid, idx in self.user_id_map.items()}
        self.num_users = len(self.selected_original_user_ids)

        all_interactions = self._load_interactions(self.selected_original_user_ids)
        self._all_interactions = all_interactions.copy()
        interactions = self._apply_history_cutoff(all_interactions)
        self._interactions = interactions.copy()

        # Restrict the online serving catalog to items visible up to the cutoff
        # so offline evaluation does not leak future-only items.
        unique_item_ids = sorted(interactions["video_id"].unique().tolist())
        self.item_id_map = {vid: idx for idx, vid in enumerate(unique_item_ids)}
        self.inverse_item_id_map = {idx: vid for vid, idx in self.item_id_map.items()}
        self.num_items = len(unique_item_ids)

        item_metadata = self._load_item_metadata(unique_item_ids)
        self._build_item_semantic_embeddings(unique_item_ids, item_metadata)
        self._build_user_state_embeddings(user_inference_rows)
        self._build_user_features()
        self._build_item_features(item_metadata)
        self._build_user_histories(interactions)
        self._build_hot_items(interactions)

    def _apply_history_cutoff(self, interactions: pd.DataFrame) -> pd.DataFrame:
        if not self.max_timestamp_by_user:
            return interactions

        cutoff_series = interactions["user_id"].map(self.max_timestamp_by_user)
        keep_mask = cutoff_series.isna() | (interactions["timestamp"] <= cutoff_series)
        filtered = interactions[keep_mask].copy()
        filtered = filtered.sort_values(["user_id", "timestamp"])
        return filtered

    def _load_user_inferences(self) -> Dict[int, Dict]:
        rows: Dict[int, Dict] = {}
        with self.user_inference_file.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                obj = json.loads(line)
                user_id = _extract_row_user_id(obj)
                if user_id is None:
                    continue
                rows[user_id] = obj
        return rows

    def _load_interactions(self, selected_user_ids: List[int]) -> pd.DataFrame:
        selected_set = set(selected_user_ids)
        chunks = []
        for chunk in pd.read_csv(
            self.interaction_file,
            usecols=["user_id", "video_id", "play_duration", "video_duration", "time", "date", "timestamp", "watch_ratio"],
            chunksize=300_000,
        ):
            filtered = chunk[chunk["user_id"].isin(selected_set)]
            if not filtered.empty:
                chunks.append(filtered)
        if not chunks:
            raise RuntimeError("No interactions found for selected users.")
        interactions = pd.concat(chunks, ignore_index=True)
        interactions = interactions.dropna(subset=["time", "date", "timestamp"])
        interactions = interactions.sort_values(["user_id", "timestamp"])
        return interactions

    def _load_item_metadata(self, item_ids: List[int]) -> Dict[int, Dict]:
        item_id_set = set(item_ids)
        metadata: Dict[int, Dict] = {
            item_id: {"text": "", "category_path": "", "supplement_categories": [], "topic_tags": [], "video_duration": 0.0}
            for item_id in item_ids
        }

        # Primary text / category table
        try:
            item_text_df = pd.read_csv(
                DEFAULT_ITEM_TEXT_FILE,
                usecols=[
                    "video_id",
                    "manual_cover_text",
                    "caption",
                    "topic_tag",
                    "first_level_category_name",
                    "second_level_category_name",
                    "third_level_category_name",
                ],
                on_bad_lines="skip",
                engine="python",
            )
        except Exception as error:
            raise RuntimeError(f"Failed to load item text file: {error}") from error

        item_text_df["video_id"] = pd.to_numeric(item_text_df["video_id"], errors="coerce")
        item_text_df = item_text_df.dropna(subset=["video_id"]).copy()
        item_text_df["video_id"] = item_text_df["video_id"].astype(int)
        item_text_df = item_text_df[item_text_df["video_id"].isin(item_id_set)]

        for _, row in item_text_df.iterrows():
            item_id = int(row["video_id"])
            caption = _clean_text(row.get("caption"))
            manual_cover = _clean_text(row.get("manual_cover_text"))
            category_parts = [
                _clean_text(row.get("first_level_category_name")),
                _clean_text(row.get("second_level_category_name")),
                _clean_text(row.get("third_level_category_name")),
            ]
            category_path = " > ".join([part for part in category_parts if part])
            metadata[item_id]["text"] = caption or manual_cover
            metadata[item_id]["category_path"] = category_path
            metadata[item_id]["topic_tags"] = _parse_topic_tags(row.get("topic_tag"))

        # Supplementary raw categories
        raw_cat_df = pd.read_csv(
            DEFAULT_ITEM_RAW_CATEGORY_FILE,
            usecols=["video_id", "category_name", "prob", "category_online", "root_name", "parent_name"],
        )
        raw_cat_df = raw_cat_df[raw_cat_df["category_online"] == 1]
        raw_cat_df = raw_cat_df[raw_cat_df["video_id"].isin(item_id_set)]

        for item_id, group in raw_cat_df.groupby("video_id"):
            top_rows = group.nlargest(3, "prob")
            supplement_paths = []
            for _, row in top_rows.iterrows():
                path_parts = [
                    _clean_text(row.get("root_name")),
                    _clean_text(row.get("parent_name")),
                    _clean_text(row.get("category_name")),
                ]
                path_parts = [part for part in path_parts if part]
                if path_parts:
                    supplement_paths.append(" > ".join(dict.fromkeys(path_parts)))
            metadata[int(item_id)]["supplement_categories"] = supplement_paths

        # Daily stats for video_duration
        item_daily_df = pd.read_csv(
            DEFAULT_ITEM_DAILY_FILE,
            usecols=["video_id", "date", "video_duration"],
        )
        item_daily_df = item_daily_df[item_daily_df["video_id"].isin(item_id_set)]
        item_daily_df = item_daily_df.sort_values(["video_id", "date"]).drop_duplicates("video_id", keep="last")
        for _, row in item_daily_df.iterrows():
            item_id = int(row["video_id"])
            metadata[item_id]["video_duration"] = float(row.get("video_duration") or 0.0)

        return metadata

    def _build_item_semantic_embeddings(self, item_ids: List[int], item_metadata: Dict[int, Dict]):
        texts = []
        for item_id in item_ids:
            meta = item_metadata[item_id]
            parts = [
                meta["text"],
                meta["category_path"],
                " ".join(meta["supplement_categories"]),
                " ".join(meta["topic_tags"]),
            ]
            text = "\n".join([part for part in parts if part])
            texts.append(text or f"video_{item_id}")

        embeddings = self.text_encoder.encode(texts)
        self._item_semantic_embs = embeddings

        for original_item_id in item_ids:
            local_item_id = self.item_id_map[original_item_id]
            meta = item_metadata[original_item_id]
            combined_text = "\n".join(
                [part for part in [meta["text"], meta["category_path"], " ".join(meta["supplement_categories"]), " ".join(meta["topic_tags"])] if part]
            )
            self._item_texts_by_local_id[local_item_id] = combined_text
            self._item_category_names_by_local_id[local_item_id] = meta["category_path"] or (
                meta["supplement_categories"][0] if meta["supplement_categories"] else "unknown"
            )

    def _build_user_state_embeddings(self, user_inference_rows: Dict[int, Dict]):
        user_state_texts: List[str] = []
        for original_user_id in self.selected_original_user_ids:
            row = user_inference_rows[original_user_id]
            compact_fields = compact_state_texts_from_any(row, fallback_user_id=original_user_id)
            user_state_texts.extend(
                [
                    _clean_text(compact_fields.get(COMPACT_FIELD_NAMES[0])) or f"user_{original_user_id}_long_term",
                    _clean_text(compact_fields.get(COMPACT_FIELD_NAMES[1])) or f"user_{original_user_id}_life_stage",
                    _clean_text(compact_fields.get(COMPACT_FIELD_NAMES[2])) or f"user_{original_user_id}_psychological",
                    _clean_text(compact_fields.get(COMPACT_FIELD_NAMES[3])) or f"user_{original_user_id}_retrieval",
                    _clean_text(compact_fields.get(COMPACT_FIELD_NAMES[4])) or f"user_{original_user_id}_growth",
                ]
            )

        embeddings = self.text_encoder.encode(user_state_texts)
        self._user_state_embs = embeddings.reshape(self.num_users, 5, self.emb_dim)

    def _build_user_features(self):
        numeric_df = pd.read_csv(DEFAULT_USER_FEATURES_FILE)
        numeric_df = numeric_df[numeric_df["user_id"].isin(self.selected_original_user_ids)]
        numeric_df = numeric_df.set_index("user_id")

        raw_df = pd.read_csv(DEFAULT_USER_FEATURES_RAW_FILE)
        raw_df = raw_df[raw_df["user_id"].isin(self.selected_original_user_ids)]
        raw_df = raw_df.set_index("user_id")

        numeric_feature_cols = [
            "is_lowactive_period",
            "is_live_streamer",
            "is_video_author",
            "follow_user_num",
            "fans_user_num",
            "friend_user_num",
            "register_days",
        ] + [f"onehot_feat{i}" for i in range(18)]

        for original_user_id in self.selected_original_user_ids:
            local_user_id = self.user_id_map[original_user_id]
            raw_row = raw_df.loc[original_user_id] if original_user_id in raw_df.index else None
            numeric_row = numeric_df.loc[original_user_id] if original_user_id in numeric_df.index else None

            if numeric_row is not None:
                numeric_values = pd.to_numeric(numeric_row[numeric_feature_cols], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            else:
                numeric_values = np.zeros(len(numeric_feature_cols), dtype=np.float32)

            if len(numeric_values) < 32:
                numeric_values = np.pad(numeric_values, (0, 32 - len(numeric_values)))
            else:
                numeric_values = numeric_values[:32]

            self._user_features[local_user_id] = UserFeatures(
                user_id=local_user_id,
                age=_age_range_to_midpoint(raw_row["age_range"]) if raw_row is not None else 28,
                gender=_gender_to_int(raw_row["gender"]) if raw_row is not None else 0,
                user_active_degree=_active_degree_to_float(raw_row["user_active_degree"]) if raw_row is not None else 0.5,
                feature_vector=numeric_values.astype(np.float32),
            )

    def _build_item_features(self, item_metadata: Dict[int, Dict]):
        category_name_to_id: Dict[str, int] = {}

        for original_item_id, meta in item_metadata.items():
            local_item_id = self.item_id_map[original_item_id]
            category_path = meta["category_path"] or (meta["supplement_categories"][0] if meta["supplement_categories"] else "unknown")
            first_category = category_path.split(" > ")[0] if category_path else "unknown"
            if first_category not in category_name_to_id:
                category_name_to_id[first_category] = len(category_name_to_id)
            category_id = category_name_to_id[first_category]

            topic_tags = meta["topic_tags"][:5]
            reduced_vector = self._item_semantic_embs[local_item_id].reshape(32, -1).mean(axis=1).astype(np.float32)

            self._item_features[local_item_id] = ItemFeatures(
                item_id=local_item_id,
                category_id=category_id,
                tags=topic_tags,
                video_duration=int(0 if pd.isna(meta.get("video_duration")) else meta.get("video_duration") or 0),
                feature_vector=reduced_vector,
            )

    def _build_user_histories(self, interactions: pd.DataFrame):
        for original_user_id, group in interactions.groupby("user_id"):
            local_user_id = self.user_id_map[int(original_user_id)]
            history_item_ids = [self.item_id_map[int(video_id)] for video_id in group["video_id"].tolist()]
            history_timestamps = [int(ts) for ts in group["timestamp"].tolist()]
            self._user_history[local_user_id] = UserHistory(
                user_id=local_user_id,
                history_item_ids=history_item_ids,
                history_timestamps=history_timestamps,
            )

    def _build_hot_items(self, interactions: pd.DataFrame):
        counts = interactions["video_id"].value_counts()
        self._hot_items_cache = [self.item_id_map[int(video_id)] for video_id in counts.index if int(video_id) in self.item_id_map]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_user_features(self, user_id: int) -> UserFeatures:
        return self._user_features[user_id]

    def get_item_features(self, item_id: int) -> ItemFeatures:
        return self._item_features[item_id]

    def get_user_history(self, user_id: int, max_len: int = 20) -> List[int]:
        history = self._user_history[user_id].history_item_ids
        return history[-max_len:] if len(history) > max_len else history

    def get_user_state_embs(self, user_id: int) -> np.ndarray:
        return self._user_state_embs[user_id]

    def get_item_semantic_embs(self, item_id: int) -> np.ndarray:
        return self._item_semantic_embs[item_id]

    def get_batch_user_state_embs(self, user_ids: List[int]) -> np.ndarray:
        return self._user_state_embs[user_ids]

    def get_batch_item_semantic_embs(self, item_ids: List[int]) -> np.ndarray:
        return self._item_semantic_embs[item_ids]

    def load_item_semantic_embs(self) -> np.ndarray:
        return self._item_semantic_embs

    def load_user_state_embs(self) -> np.ndarray:
        return self._user_state_embs

    def load_item_co_occurrence(self) -> Dict[Tuple[int, int], int]:
        if self._cooccurrence_cache is not None:
            return self._cooccurrence_cache

        co_occurrence: Dict[Tuple[int, int], int] = {}
        for user_history in self._user_history.values():
            recent = user_history.history_item_ids[-self.cooccurrence_history_limit :]
            unique_recent = list(dict.fromkeys(recent))
            for index, item_i in enumerate(unique_recent):
                for item_j in unique_recent[index + 1 :]:
                    key = (min(item_i, item_j), max(item_i, item_j))
                    co_occurrence[key] = co_occurrence.get(key, 0) + 1

        self._cooccurrence_cache = co_occurrence
        return co_occurrence

    def load_hot_items(self) -> List[int]:
        return self._hot_items_cache or []

    def load_user_states(self) -> Dict[int, Dict[str, np.ndarray]]:
        user_states = {}
        for local_user_id in range(self.num_users):
            user_states[local_user_id] = {
                USER_STATE_FIELDS[0]: self._user_state_embs[local_user_id][0],
                USER_STATE_FIELDS[1]: self._user_state_embs[local_user_id][1],
                USER_STATE_FIELDS[2]: self._user_state_embs[local_user_id][2],
                USER_STATE_FIELDS[3]: self._user_state_embs[local_user_id][3],
                USER_STATE_FIELDS[4]: self._user_state_embs[local_user_id][4],
            }
        return user_states

    def get_statistics(self) -> Dict[str, float]:
        history_lengths = [len(history.history_item_ids) for history in self._user_history.values()]
        return {
            "num_users": self.num_users,
            "num_items": self.num_items,
            "embedding_dimension": self.emb_dim,
            "avg_history_length": float(np.mean(history_lengths)) if history_lengths else 0.0,
            "min_history_length": float(np.min(history_lengths)) if history_lengths else 0.0,
            "max_history_length": float(np.max(history_lengths)) if history_lengths else 0.0,
        }

    def original_user_id(self, local_user_id: int) -> int:
        return self.inverse_user_id_map[local_user_id]

    def original_item_id(self, local_item_id: int) -> int:
        return self.inverse_item_id_map[local_item_id]

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"KuaiRecDataLoader(num_users={stats['num_users']}, "
            f"num_items={stats['num_items']}, "
            f"emb_dim={stats['embedding_dimension']}, "
            f"text_encoder={self.text_encoder_backend}, "
            f"avg_history_length={stats['avg_history_length']:.2f})"
        )
