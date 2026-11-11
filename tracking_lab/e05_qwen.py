"""Local E05 Qwen adapter with labelled images, attribute QA, and call accounting.

The QA formulation borrows the contrastive-attribute idea from MSRTrack. It is
an explicitly different local-model adaptation, not the original paper method.
Importing this module does not import Torch, Transformers, Pillow, or Qwen tools.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import threading
import time


PROMPT_VERSION = "e05-labelled-attribute-qa-v1"
STATES = frozenset({"supported", "contradicted", "unobservable"})
LETTERS = frozenset("ABCD")

PROPOSER_PROMPT = (
    "Identify at most two persistent, directly visible attributes that distinguish "
    "the initialization target from supplied possible confusing proposals. These "
    "proposals may contain the same target, partial objects or background, so do "
    "not assume that each is a different individual. Images "
    "are explicitly labelled. The target is the same individual to be tracked, not "
    "merely a category. Avoid image position, background, transient pose, inferred "
    "motion, physical measurements, or details too small to see. Do not invent a "
    "difference when objects cannot be distinguished. Judge the object inside the "
    "blue proposal rectangle, not any object elsewhere in the context crop. Return only JSON with one "
    "key features, a list of at most two objects with exactly these keys: attribute "
    "(string), target_value (string), distractor_values (list of strings). Use an "
    "empty list when there is no defensible distinguishing attribute."
)
DIRECT_PROMPT = (
    "Verify the identity of the original initialization target among current visual "
    "tracker proposals. Each image has an explicit reference/history/distractor/"
    "candidate label immediately before it. Past crops are predictions, not verified "
    "truth, and may already be wrong. Initialization distractor images are merely "
    "possible confusing proposals and may show the target itself or background. "
    "The supplied attributes are hypotheses, not verified facts. Candidate order is arbitrary. Select a supplied "
    "candidate letter only when visible evidence supports the same individual and "
    "the object inside the blue proposal rectangle, not any object elsewhere in "
    "the context crop. Use NONE when evidence supports that none matches; "
    "use ABSTAIN when identity is ambiguous or not visible. Neither rejection means "
    "the target is absent from the full scene. Do not infer unseen markings, motion "
    "or physical measurements. Return only JSON with exactly two string keys answer "
    "and evidence. answer is a supplied A-D letter, NONE or ABSTAIN; evidence gives "
    "a short visible reason."
)
QA_PROMPT = (
    "Evaluate the supplied target-identity attributes for EACH labelled current "
    "candidate, using the initialization reference and the same supplied causal "
    "history/distractors. Attributes are hypotheses and can be wrong; history is "
    "unverified tracker output. Possible initialization distractors may contain "
    "the same target, partial objects or background. Judge only the object inside "
    "each blue proposal rectangle. For each attribute return supported only if the "
    "candidate visibly has the target value; contradicted only if visible evidence "
    "conflicts with that value (including a stated distractor value); unobservable "
    "if resolution, occlusion, view or ambiguity prevents deciding. Do not guess "
    "unseen details or choose a candidate yourself. Return only JSON with exactly "
    "candidate_answers and evidence. candidate_answers maps EVERY supplied candidate "
    "letter to a list of supported/contradicted/unobservable strings, one in the "
    "same order as each supplied feature. evidence is a short visible explanation. "
    "All image roles and candidate letters are given explicitly before each image."
)


class BudgetExceeded(RuntimeError):
    pass


class GenerationFailure(RuntimeError):
    def __init__(self, message, event_id):
        super().__init__(message)
        self.event_id = event_id


class FeatureCompatibilityError(RuntimeError):
    pass


def _hash_json(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON key")
        value[key] = item
    return value


def _read_object(raw):
    value = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError("Expected one JSON object")
    return value


def _text(value, maximum=1000):
    return isinstance(value, str) and 0 < len(value.strip()) <= maximum


def validate_features(features):
    if not isinstance(features, list) or len(features) > 2:
        raise ValueError("Need zero to two attribute objects")
    for feature in features:
        if not isinstance(feature, dict) or set(feature) != {"attribute", "target_value", "distractor_values"}:
            raise ValueError("Invalid feature keys")
        if not _text(feature["attribute"], 200) or not _text(feature["target_value"], 200):
            raise ValueError("Invalid target attribute")
        negative = feature["distractor_values"]
        if not isinstance(negative, list) or len(negative) > 4 or not all(_text(x, 200) for x in negative):
            raise ValueError("Invalid distractor attributes")
    return features


def parse_features(raw):
    try:
        value = _read_object(raw)
        if set(value) != {"features"}:
            raise ValueError("Expected features key only")
        return {"status": "VALID", "features": validate_features(value["features"])}
    except (ValueError, TypeError, KeyError) as error:
        return {"status": "INVALID", "features": [], "parse_error": str(error)}


def parse_direct(raw, labels):
    try:
        value = _read_object(raw)
        if set(value) != {"answer", "evidence"} or not _text(value["evidence"]):
            raise ValueError("Invalid choice schema")
        if not isinstance(value["answer"], str) or value["answer"] not in set(labels) | {"NONE", "ABSTAIN"}:
            raise ValueError("Choice not a supplied candidate or rejection")
        return {"status": "VALID", **value}
    except (ValueError, TypeError, KeyError) as error:
        return {"status": "INVALID", "answer": "INVALID", "evidence": "", "parse_error": str(error)}


def derive_qa_choice(table):
    """Fixed program decision, never a second unconstrained model choice."""
    scores = {label: {"matches": answers.count("supported"),
                      "contradictions": answers.count("contradicted"),
                      "unobservable": answers.count("unobservable")}
              for label, answers in table.items()}
    eligible = {label: item["matches"] for label, item in scores.items()
                if item["matches"] >= 1 and item["contradictions"] == 0}
    if eligible:
        maximum = max(eligible.values())
        best = [label for label, count in eligible.items() if count == maximum]
        answer = best[0] if len(best) == 1 else "ABSTAIN"
    elif scores and all(item["contradictions"] >= 1 for item in scores.values()):
        answer = "NONE"
    else:
        answer = "ABSTAIN"
    return answer, scores


def parse_qa(raw, labels, feature_count):
    try:
        value = _read_object(raw)
        if set(value) != {"candidate_answers", "evidence"} or not _text(value["evidence"]):
            raise ValueError("Invalid attribute QA schema")
        table = value["candidate_answers"]
        if not isinstance(table, dict) or set(table) != set(labels) or feature_count not in (1, 2):
            raise ValueError("QA labels or feature count differ")
        for answers in table.values():
            if not isinstance(answers, list) or len(answers) != feature_count or not all(
                    isinstance(x, str) and x in STATES for x in answers):
                raise ValueError("Every candidate needs one state per feature")
        answer, scores = derive_qa_choice(table)
        return {"status": "VALID", "answer": answer, "evidence": value["evidence"],
                "support_table": table, "support_counts": scores,
                "decision_rule": "unique maximum supported>=1 with zero contradicted; otherwise NONE only if all contradicted, else ABSTAIN"}
    except (ValueError, TypeError, KeyError) as error:
        return {"status": "INVALID", "answer": "INVALID", "evidence": "", "parse_error": str(error)}


def _image(value):
    return value["image"] if isinstance(value, dict) else value


def _labels(candidates):
    labels = [x["label"] for x in candidates]
    if not 1 <= len(labels) <= 4 or len(set(labels)) != len(labels) or not set(labels) <= LETTERS:
        raise ValueError("Need one to four unique supplied A-D labels")
    return labels


def _append_image(content, role, image):
    content.extend([{"type": "text", "text": role}, {"type": "image", "image": _image(image)}])


def build_propose_messages(reference, distractor_images):
    content = []
    _append_image(content, "INITIALIZATION TARGET REFERENCE (original target identity):", reference)
    for index, image in enumerate(distractor_images or [], 1):
        _append_image(content, f"POSSIBLE INITIALIZATION DISTRACTOR {index} (unverified proposal; may show target or background):", image)
    return [{"role": "system", "content": [{"type": "text", "text": PROPOSER_PROMPT}]},
            {"role": "user", "content": content}]


def build_choose_messages(reference, history, candidates, features=None, mode="direct", distractors=None):
    _labels(candidates)
    if mode not in ("direct", "contrastive"):
        raise ValueError("Unknown E05 mode")
    features = validate_features(features or [])
    content = []
    _append_image(content, "INITIALIZATION TARGET REFERENCE (original target identity):", reference)
    for index, image in enumerate(distractors or [], 1):
        _append_image(content, f"POSSIBLE INITIALIZATION DISTRACTOR {index} (unverified proposal; same image supplied to every selector):", image)
    for index, image in enumerate(history or [], 1):
        _append_image(content, f"PAST TRACKER PREDICTION {index} (strictly past, unverified identity):", image)
    for candidate in candidates:
        _append_image(content, f"CURRENT CANDIDATE {candidate['label']} (arbitrary label; this image belongs to {candidate['label']}):", candidate)
    prompt = DIRECT_PROMPT if mode == "direct" else QA_PROMPT
    content.append({"type": "text", "text": "TARGET ATTRIBUTE HYPOTHESES IN ORDER (unverified; same for both output modes): " + json.dumps(features, ensure_ascii=False)})
    return [{"role": "system", "content": [{"type": "text", "text": prompt}]},
            {"role": "user", "content": content}]


def _pil(value):
    from PIL import Image
    value = _image(value)
    if isinstance(value, (str, Path)):
        with Image.open(value) as image:
            return image.convert("RGB")
    if isinstance(value, Image.Image):
        return value.convert("RGB").copy()
    raise TypeError("E05 images must be PIL images or local paths")


def _pixel_hash(image):
    header = json.dumps([image.mode, *image.size]).encode()
    return hashlib.sha256(header + image.tobytes()).hexdigest()


class CallLedger:
    """Persistent attempted-invocation budget; failed calls remain counted."""
    def __init__(self, root, maximum=50):
        if not 1 <= maximum <= 50:
            raise ValueError("E05 budget must be 1..50")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "call_ledger.jsonl"
        self.maximum = maximum
        self._thread_lock = threading.Lock()

    @property
    def used(self):
        if not self.path.exists():
            return 0
        rows = [json.loads(x) for x in self.path.read_text(encoding="utf-8").splitlines() if x.strip()]
        begun = [x["event_id"] for x in rows if x["event"] == "begin"]
        if len(begun) != len(set(begun)):
            raise ValueError("Duplicate invocation IDs in ledger")
        return len(begun)

    def append(self, value):
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def reserve(self, kind, case_id):
        if kind not in {"propose", "direct", "contrastive"}:
            raise ValueError("Unknown generation kind")
        lock = self.root / ".ledger.lock"
        with self._thread_lock:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                count = self.used
                if count >= self.maximum:
                    raise BudgetExceeded(f"E05 generation budget exhausted: {count}/{self.maximum}")
                event_id = f"{count + 1:04d}_{kind}"
                folder = self.root / "events" / event_id
                folder.mkdir(parents=True, exist_ok=False)
                self.append({"event": "begin", "event_id": event_id, "kind": kind,
                             "case_id": str(case_id), "utc": datetime.now(timezone.utc).isoformat(),
                             "attempt_number": count + 1, "maximum": self.maximum})
                return event_id, folder
            finally:
                os.close(descriptor)
                lock.unlink()


class E05Qwen:
    def __init__(self, model_path, output_dir, max_calls=50, device="cuda",
                 max_pixels=262144, max_new_tokens=192):
        self.model_path = Path(model_path).expanduser().resolve()
        self.output_dir = Path(output_dir)
        self.ledger = CallLedger(self.output_dir, max_calls)
        self.device = device
        self.max_pixels = int(max_pixels)
        self.max_new_tokens = int(max_new_tokens)
        self.model = self.processor = None
        self.metadata = {"model_path": str(self.model_path), "prompt_version": PROMPT_VERSION,
                         "local_files_only": True, "device": device, "use_fast": False,
                         "max_pixels_per_image": self.max_pixels, "max_new_tokens": self.max_new_tokens}

    @property
    def calls_used(self):
        return self.ledger.used

    def load(self):
        if self.model is not None:
            return
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        if not str(self.device).startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("Actual E05 inference requires CUDA; no silent CPU fallback")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA bfloat16 is required")
        if not self.model_path.is_dir():
            raise FileNotFoundError("Need an existing local Qwen model directory")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        self.processor = AutoProcessor.from_pretrained(str(self.model_path), local_files_only=True,
                                                       use_fast=False, max_pixels=self.max_pixels)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(self.model_path), local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(self.device).eval()
        actual_type = type(self.model.config).model_type
        if actual_type != "qwen2_5_vl":
            raise RuntimeError("Loaded model is not Qwen2.5-VL")
        self.metadata.update({"transformers": transformers.__version__, "torch": torch.__version__,
                              "cuda_runtime": torch.version.cuda,
                              "model_class": type(self.model).__name__, "dtype": "bfloat16", "attention": "sdpa",
                              "actual_model_commit": getattr(self.model.config, "_commit_hash", None),
                              "config_sha256": hashlib.sha256((self.model_path / "config.json").read_bytes()).hexdigest(),
                              "processor_class": type(self.processor).__name__})
        self._write(self.output_dir / "model_metadata.json", self.metadata)

    @staticmethod
    def _write(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def _materialize(self, messages, folder):
        actual, metadata = [], []
        index = 0
        for message in messages:
            content, descriptions = [], []
            previous_text = None
            for block in message["content"]:
                if block["type"] == "text":
                    previous_text = block["text"]
                    content.append(dict(block)); descriptions.append(dict(block))
                elif block["type"] == "image":
                    if not previous_text:
                        raise ValueError("Every image needs its explicit role text immediately before it")
                    image = _pil(block["image"])
                    name = f"image_{index:02d}.png"; index += 1
                    image.save(folder / name)
                    content.append({"type": "image", "image": image})
                    descriptions.append({"type": "image", "saved_file": name, "role_text": previous_text,
                                         "pixel_sha256": _pixel_hash(image), "size_wh": list(image.size)})
                    previous_text = None
                else:
                    raise ValueError("Unknown message content type")
            actual.append({"role": message["role"], "content": content})
            metadata.append({"role": message["role"], "content": descriptions})
        self._write(folder / "messages.json", {"messages": metadata, "message_sha256": _hash_json(metadata),
                                               "model": self.metadata})
        return actual

    def _model_generate(self, messages, folder):
        self.load()
        import torch
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Inputs are already materialized RGB PIL images. Passing them directly
        # uses the same image-processor path as the visual-feature control,
        # without an extra qwen_vl_utils pre-resize in only the generation arm.
        image_inputs = [block["image"] for message in messages for block in message["content"]
                        if block["type"] == "image"]
        inputs = self.processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(self.device)
        self._write(folder / "processor_inputs.json", {
            "image_grid_thw": inputs["image_grid_thw"].detach().cpu().tolist(),
            "input_ids_sha256": hashlib.sha256(inputs["input_ids"].detach().cpu().numpy().tobytes()).hexdigest(),
            "input_tokens": int(inputs["input_ids"].shape[1]),
            "pixel_values_shape": list(inputs["pixel_values"].shape),
            "pixel_values_sha256": hashlib.sha256(inputs["pixel_values"].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
            "image_count": len(image_inputs), "do_sample": False,
            "max_new_tokens": self.max_new_tokens})
        torch.cuda.synchronize()
        with torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens)
        torch.cuda.synchronize()
        return self.processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    def _invoke(self, kind, case_id, messages, parser):
        event_id, folder = self.ledger.reserve(kind, case_id)
        started = time.perf_counter()
        invoked = False
        try:
            actual = self._materialize(messages, folder)
            invoked = True
            raw = self._model_generate(actual, folder)
            parsed = parser(raw)
            result = {**parsed, "raw": raw, "event_id": event_id,
                      "seconds": time.perf_counter() - started, "generation_calls": 1,
                      "kind": kind, "case_id": str(case_id)}
            self._write(folder / "result.json", result)
            self.ledger.append({"event": "end", "event_id": event_id, "status": result["status"],
                                "seconds": result["seconds"], "model_generate_entry_invoked": invoked})
            return result
        except Exception as error:
            result = {"status": "ERROR", "event_id": event_id, "kind": kind,
                      "case_id": str(case_id), "error_class": type(error).__name__,
                      "error": str(error)[:1000], "seconds": time.perf_counter() - started,
                      "model_generate_entry_invoked": invoked, "generation_calls": 1}
            self._write(folder / "result.json", result)
            self.ledger.append({"event": "end", "event_id": event_id, "status": "ERROR",
                                "error_class": type(error).__name__, "seconds": result["seconds"]})
            raise GenerationFailure(f"E05 generation failed; inspect {folder / 'result.json'}", event_id) from error

    def propose(self, reference, distractor_images=None, case_id="initialization"):
        return self._invoke("propose", case_id,
                            build_propose_messages(reference, distractor_images or []), parse_features)

    def choose(self, reference, history, candidates, features=None, mode="direct", case_id="query", distractors=None):
        labels = _labels(candidates)
        features = validate_features(features or [])
        if mode == "contrastive" and not features:
            return {"status": "NOT_RUN", "answer": "ABSTAIN", "evidence": "No accepted identity features.",
                    "generation_calls": 0, "kind": mode, "case_id": str(case_id)}
        messages = build_choose_messages(reference, history, candidates, features, mode, distractors)
        parser = (lambda raw: parse_direct(raw, labels)) if mode == "direct" else (
            lambda raw: parse_qa(raw, labels, len(features)))
        return self._invoke(mode, case_id, messages, parser)

    def _visual_vector(self, image):
        self.load()
        import torch
        tensors = self.processor.image_processor(images=[_pil(image)], max_pixels=self.max_pixels,
                                                 return_tensors="pt").to(self.device)
        with torch.inference_mode():
            if callable(getattr(self.model, "get_image_features", None)):
                output = self.model.get_image_features(pixel_values=tensors["pixel_values"],
                                                       image_grid_thw=tensors["image_grid_thw"])
                if isinstance(output, (tuple, list)) and len(output) == 1:
                    output = output[0]
                api = "get_image_features"
            elif callable(getattr(self.model, "visual", None)):
                output = self.model.visual(tensors["pixel_values"].to(dtype=self.model.visual.dtype),
                                           grid_thw=tensors["image_grid_thw"])
                api = "visual_legacy_4_51"
            else:
                raise FeatureCompatibilityError("No supported Qwen2.5 visual API")
            if not isinstance(output, torch.Tensor) or output.ndim != 2 or output.shape[0] < 1:
                raise FeatureCompatibilityError("Expected one merged visual token matrix [tokens,width]")
            vector = output.float().mean(dim=0)
            norm = torch.linalg.vector_norm(vector)
            if not torch.isfinite(vector).all().item() or not torch.isfinite(norm).item() or norm.item() <= 0:
                raise FeatureCompatibilityError("Invalid merged visual vector")
            self.metadata["visual_feature_api"] = api
            return vector / norm

    def visual_scores(self, reference, history, candidates, distractors=None):
        _labels(candidates)
        self.load()
        import torch
        history = list(history or [])
        distractors = list(distractors or [])
        started = time.perf_counter()
        vectors = [self._visual_vector(reference)] + [self._visual_vector(x) for x in history]
        pooled = torch.stack(vectors).mean(0)
        norm = torch.linalg.vector_norm(pooled)
        if not torch.isfinite(norm).item() or norm.item() <= 0:
            raise FeatureCompatibilityError("Invalid mean_history reference vector")
        pooled = pooled / norm
        negatives = [self._visual_vector(x) for x in distractors]
        scores, contrastive_scores = {}, {}
        for candidate in candidates:
            vector = self._visual_vector(candidate)
            positive = float((pooled @ vector).item())
            scores[candidate["label"]] = positive
            contrastive_scores[candidate["label"]] = positive - max(
                (float((negative @ vector).item()) for negative in negatives), default=0.0)
        torch.cuda.synchronize()
        return {"status": "VALID", "scores": scores, "contrastive_scores": contrastive_scores,
                "answer": max(scores, key=scores.get), "contrastive_answer": max(contrastive_scores, key=contrastive_scores.get),
                "memory_mode": "mean_history", "history_images_used": len(history),
                "negative_images_used": len(negatives), "negative_weight": 1.0,
                "image_encoder_calls": len(vectors) + len(negatives) + len(candidates),
                "seconds": time.perf_counter() - started, "feature_api": self.metadata["visual_feature_api"],
                "image_pixel_hashes": {"reference": _pixel_hash(_pil(reference)),
                                       "history": [_pixel_hash(_pil(x)) for x in history],
                                       "distractors": [_pixel_hash(_pil(x)) for x in distractors],
                                       "candidates": {x["label"]: _pixel_hash(_pil(x)) for x in candidates}}}
