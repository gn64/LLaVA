from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import re
from torchvision.ops import box_iou
import torch

from torchmetrics.detection.mean_ap import MeanAveragePrecision


def extract_bounding_boxes(answer):
    """Extract bounding boxes from the answer string."""
    pattern = r"\[([\d\.]+),\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\]"
    return [list(map(float, match)) for match in re.findall(pattern, answer)]


def evaluate_boxes(output_list, iou_threshold=0.5):
    """Evaluate the model's predicted bounding boxes against ground truth boxes.

    Args:
        output_list (list of dicts): List of dicts containing model outputs and actual bounding boxes.

    Returns:
        dict: A dictionary containing performance metrics (Precision, Recall, F1 Score).
    """
    # Initialize MeanAveragePrecision metric
    metric = MeanAveragePrecision(iou_thresholds=[iou_threshold])

    for output_single in output_list:
        if not ("output" in output_single and "boxes" in output_single):
            raise ValueError(
                "Both keys 'output' and 'boxes' must be contained in dict."
            )

        output_text = output_single["output"]
        predicted_boxes = extract_bounding_boxes(output_text)
        actual_boxes = output_single["boxes"]

        if predicted_boxes and actual_boxes:
            pred_boxes_tensor = torch.tensor(predicted_boxes, dtype=torch.float32)
            gt_boxes_tensor = torch.tensor(actual_boxes, dtype=torch.float32)

            # Assuming all boxes belong to the same category (category_id = 0)
            metric.update(
                preds=[
                    {
                        "boxes": pred_boxes_tensor,
                        "scores": torch.ones(len(pred_boxes_tensor)),
                        "labels": torch.zeros(
                            len(pred_boxes_tensor), dtype=torch.int64
                        ),
                    }
                ],
                target=[
                    {
                        "boxes": gt_boxes_tensor,
                        "labels": torch.zeros(len(gt_boxes_tensor), dtype=torch.int64),
                    }
                ],
            )

    # Compute metrics
    results = metric.compute()

    metrics = {
        "Precision(macro)": results["map"].item(),
        "Recall(macro)": results["mar_100"].item(),
        "F1 Score(macro)": results[
            "map"
        ].item(),  # F1 score isn't directly available; using mAP as a proxy here
    }

    return metrics


def process_match_classification_with_metrics(output_list, labels=None):
    """Process the data to match classification with output and calculate metrics.
    Args:
        output_list (list of dicts): List of dicts containing model outputs and actual labels.
        labels (list): List of all possible labels.
    Returns:
        dict: A dictionary containing processed data and performance metrics (Accuracy, F1 Score).
    """
    ret = []
    predicted = []
    actual = []
    for output_single in output_list:
        if not ("output" in output_single and "labels" in output_single):
            raise ValueError(
                "Both keys 'output' and 'labels' must be contained in dict."
            )

        ret_cell = output_single.copy()
        output_text = output_single["output"].lower()
        if labels is None:
            if "pathologies" in output_single:
                labels = output_single["pathologies"]
            else:
                raise ValueError("Labels must be provided.")
        predicted_labels = [label for label in labels if label.lower() in output_text]
        actual_labels = [label.lower() for label in output_single["labels"]]

        correct_labels = [label for label in predicted_labels if label in actual_labels]
        incorrect_labels = [
            label for label in predicted_labels if label not in actual_labels
        ]
        missing_labels = [
            label for label in actual_labels if label not in predicted_labels
        ]
        predicted_arrays = [
            1 if label.lower() in output_text else 0 for label in labels
        ]
        actual_arrays = [1 if label.lower() in actual_labels else 0 for label in labels]
        predicted.append(predicted_arrays)
        actual.append(actual_arrays)
        ret_cell["correct_labels"] = correct_labels
        ret_cell["incorrect_labels"] = incorrect_labels
        ret_cell["missing_labels"] = missing_labels
        ret.append(ret_cell)
    predicted = np.array(predicted)
    actual = np.array(actual)
    precision_micro, recall_micro, f1_score_micro, _ = precision_recall_fscore_support(
        actual, predicted, average="micro"
    )
    precision_macro, recall_macro, f1_score_macro, _ = precision_recall_fscore_support(
        actual, predicted, average="macro"
    )
    metrics = {
        "Precision(macro)": precision_macro,
        "Precision(micro)": precision_micro,
        "Recall(macro)": recall_macro,
        "Recall(micro)": recall_micro,
        "F1 Score(macro)": f1_score_macro,
        "F1 Score(micro)": f1_score_micro,
    }

    return {"results": ret, "metrics": metrics}
