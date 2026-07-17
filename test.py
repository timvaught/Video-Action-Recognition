"""Compatibility helpers for external evaluation scripts."""

from sklearn.metrics import classification_report, multilabel_confusion_matrix

from train import evaluate


def get_test_report(targets, outputs, target_names):
    """Return a per-class classification report."""
    return classification_report(
        targets,
        outputs,
        labels=list(range(len(target_names))),
        output_dict=True,
        target_names=target_names,
        zero_division=0,
    )


def get_confusion_matrix(targets, outputs, labels_dict, all_categories):
    """Return one-vs-rest confusion matrices keyed by class name."""
    inverse = {label: category for category, label in labels_dict.items()}
    target_categories = [inverse[target] for target in targets]
    output_categories = [inverse[output] for output in outputs]
    matrices = multilabel_confusion_matrix(
        target_categories, output_categories, labels=all_categories
    )
    return dict(zip(all_categories, matrices))


__all__ = ["evaluate", "get_confusion_matrix", "get_test_report"]
