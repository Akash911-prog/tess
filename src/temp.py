import json
from rich import print

with open("results.json", "r", encoding="utf-8") as f:
    reports = json.load(f)

def compare_lcn_results(report):
    """
    Compare two LCNTester runs (old vs new).

    Returns a structured dict with improvements and regressions.
    """

    old_metrics = report[2]
    new_metrics = report[0]

    report = {
        "overall": {},
        "per_intent": {},
        "improved_intents": [],
        "regressed_intents": [],
        "unchanged_intents": []
    }

    # -----------------------------
    # Overall comparison
    # -----------------------------
    report["overall"] = {
        "accuracy_change": new_metrics["accuracy"] - old_metrics["accuracy"],
        "avg_confidence_change": (
            new_metrics["avg_confidence"] - old_metrics["avg_confidence"]
        ),
        "avg_latency_ms_change": (
            new_metrics["avg_latency_ms"] - old_metrics["avg_latency_ms"]
        ),
        "ambiguity_accuracy_change": (
            None if old_metrics["ambiguity_accuracy"] is None
            else new_metrics["ambiguity_accuracy"] - old_metrics["ambiguity_accuracy"]
        )
    }

    # -----------------------------
    # Per-intent accuracy
    # -----------------------------
    old_per_intent = old_metrics["per_intent_accuracy"]
    new_per_intent = new_metrics["per_intent_accuracy"]

    all_intents = set(old_per_intent) | set(new_per_intent)

    for intent in sorted(all_intents):
        old_acc = old_per_intent.get(intent, 0.0)
        new_acc = new_per_intent.get(intent, 0.0)
        delta = new_acc - old_acc

        report["per_intent"][intent] = {
            "mini_accuracy": old_acc,
            "mpnet_accuracy": new_acc,
            "delta": delta
        }

        if delta > 0.01:
            report["improved_intents"].append(intent)
        elif delta < -0.01:
            report["regressed_intents"].append(intent)
        else:
            report["unchanged_intents"].append(intent)

    return report

if __name__ == "__main__":
    print(compare_lcn_results(reports))