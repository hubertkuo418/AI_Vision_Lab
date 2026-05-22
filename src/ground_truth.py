import json


def _bbox_area(bbox):
    return max(bbox["width"], 0) * max(bbox["height"], 0)


def _intersection_area(box_a, box_b):
    x1 = max(box_a["x"], box_b["x"])
    y1 = max(box_a["y"], box_b["y"])
    x2 = min(box_a["x"] + box_a["width"], box_b["x"] + box_b["width"])
    y2 = min(box_a["y"] + box_a["height"], box_b["y"] + box_b["height"])
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    return width * height


def box_iou(box_a, box_b):
    intersection = _intersection_area(box_a, box_b)
    if intersection == 0:
        return 0.0

    union = _bbox_area(box_a) + _bbox_area(box_b) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def parse_ground_truth_payload(payload):
    """Parse GT JSON as {filename: [boxes]} or a single list of boxes."""
    if isinstance(payload, list):
        return {"default": payload}

    if not isinstance(payload, dict):
        raise ValueError("Ground truth must be a JSON object or list.")

    normalized = {}
    for key, value in payload.items():
        if not isinstance(value, list):
            raise ValueError(f"Ground truth for '{key}' must be a list of boxes.")
        normalized[key] = value
    return normalized


def parse_ground_truth_file(uploaded_file):
    payload = json.loads(uploaded_file.getvalue().decode("utf-8"))
    return parse_ground_truth_payload(payload)


def _normalize_prediction(annotation):
    bbox = annotation["bbox"]
    return {
        "label": annotation.get("label"),
        "x": int(bbox["x"]),
        "y": int(bbox["y"]),
        "width": int(bbox["width"]),
        "height": int(bbox["height"]),
    }


def _normalize_ground_truth(box):
    required = {"label", "x", "y", "width", "height"}
    missing = required - set(box.keys())
    if missing:
        raise ValueError(f"Ground-truth box missing fields: {sorted(missing)}")

    return {
        "label": box["label"],
        "x": int(box["x"]),
        "y": int(box["y"]),
        "width": int(box["width"]),
        "height": int(box["height"]),
    }


def match_predictions(predictions, ground_truth, iou_threshold=0.5):
    predictions = [_normalize_prediction(item) for item in predictions]
    ground_truth = [_normalize_ground_truth(item) for item in ground_truth]

    matched_predictions = set()
    matched_ground_truth = set()
    true_positives = 0

    for pred_index, prediction in enumerate(predictions):
        best_iou = 0.0
        best_gt_index = None

        for gt_index, gt_box in enumerate(ground_truth):
            if gt_index in matched_ground_truth:
                continue
            if prediction["label"] != gt_box["label"]:
                continue

            score = box_iou(prediction, gt_box)
            if score > best_iou:
                best_iou = score
                best_gt_index = gt_index

        if best_gt_index is not None and best_iou >= iou_threshold:
            true_positives += 1
            matched_predictions.add(pred_index)
            matched_ground_truth.add(best_gt_index)

    false_positives = len(predictions) - len(matched_predictions)
    false_negatives = len(ground_truth) - len(matched_ground_truth)
    return true_positives, false_positives, false_negatives


def classification_scores(true_positives, false_positives, false_negatives):
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def evaluate_predictions(predictions, ground_truth, iou_threshold=0.5):
    counts = match_predictions(predictions, ground_truth, iou_threshold=iou_threshold)
    return classification_scores(*counts)


def aggregate_evaluations(evaluations):
    """Micro-average precision/recall/F1 across all images."""
    if not evaluations:
        return None

    total_tp = sum(item["true_positives"] for item in evaluations)
    total_fp = sum(item["false_positives"] for item in evaluations)
    total_fn = sum(item["false_negatives"] for item in evaluations)
    return classification_scores(total_tp, total_fp, total_fn)


def macro_average_evaluations(evaluations):
    """Macro-average per-image precision/recall/F1."""
    if not evaluations:
        return None

    return {
        "precision": round(mean_score(evaluations, "precision"), 4),
        "recall": round(mean_score(evaluations, "recall"), 4),
        "f1": round(mean_score(evaluations, "f1"), 4),
    }


def mean_score(evaluations, key):
    return sum(item[key] for item in evaluations) / len(evaluations)


def build_ground_truth_report(image_names, gt_data, require_detection_gt=False):
    """Summarize which uploaded images have ground-truth labels."""
    if gt_data is None:
        return {
            "has_ground_truth": False,
            "matched_images": [],
            "missing_gt_images": list(image_names) if require_detection_gt else [],
            "unused_gt_keys": [],
            "image_count": len(image_names),
            "matched_count": 0,
        }

    matched_images = []
    missing_gt_images = []
    image_set = set(image_names)

    for file_name in image_names:
        boxes = []
        if file_name in gt_data:
            boxes = gt_data[file_name]
        elif "default" in gt_data:
            boxes = gt_data["default"]
        if boxes:
            matched_images.append(file_name)
        elif require_detection_gt:
            missing_gt_images.append(file_name)

    unused_gt_keys = [
        key
        for key in gt_data.keys()
        if key != "default" and key not in image_set
    ]

    return {
        "has_ground_truth": True,
        "matched_images": matched_images,
        "missing_gt_images": missing_gt_images,
        "unused_gt_keys": unused_gt_keys,
        "image_count": len(image_names),
        "matched_count": len(matched_images),
    }


def ground_truth_template_json():
    return json.dumps(
        {
            "photo.jpg": [
                {"label": "face", "x": 120, "y": 80, "width": 64, "height": 64}
            ],
            "default": [
                {"label": "person", "x": 40, "y": 30, "width": 120, "height": 200}
            ],
        },
        indent=2,
    )
