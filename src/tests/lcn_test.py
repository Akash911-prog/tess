import json
import time
import statistics
from collections import defaultdict


class LCNTester:
    def __init__(self, lcn):
        self.lcn = lcn
        self.test_cases = []
        self.results = []
        self.metrics = {}

    # -------------------------------------------------
    # Load test cases (NEW STRUCTURE)
    # -------------------------------------------------
    def load_test_cases(self, filepath):
        """
        Expected JSON structure:
        {
          "open": [ {...}, {...} ],
          "code": [ {...} ],
          "ambiguous": [ {...} ]
        }
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for intent_key, cases in data.items():
            for case in cases:
                self.test_cases.append({
                    "input": case["input"],
                    "expected": case.get("expected", intent_key),
                    "min_confidence": case.get("min_confidence", 0.0),
                    "should_be_ambiguous": case.get("should_be_ambiguous", False),
                    "group": intent_key
                })

    # -------------------------------------------------
    # Run tests
    # -------------------------------------------------
    def run_tests(self):
        for case in self.test_cases:
            text = case["input"]

            start = time.perf_counter()
            result = self.lcn.normalize(text)   # IntentResult
            latency = (time.perf_counter() - start) * 1000

            predicted_intent = result.intent
            confidence = result.confidence
            is_ambiguous = result.is_ambiguous

            intent_correct = predicted_intent == case["expected"]
            confidence_ok = confidence >= case["min_confidence"]

            if case["should_be_ambiguous"]:
                correct = is_ambiguous
            else:
                correct = intent_correct and confidence_ok

            self.results.append({
                "input": text,
                "expected": case["expected"],
                "predicted": predicted_intent,
                "confidence": confidence,
                "min_confidence": case["min_confidence"],
                "confidence_ok": confidence_ok,
                "correct": correct,
                "latency_ms": latency,
                "group": case["group"],

                # IntentResult details
                "ambiguous_expected": case["should_be_ambiguous"],
                "ambiguous_predicted": is_ambiguous,
                "match_quality": result.match_quality,
                "semantic_score": result.semantic_score,
                "lexical_score": result.lexical_score,
                "overlap_boost": result.overlap_boost,
                "second_best_intent": result.second_best_intent,
                "second_best_score": result.second_best_score
            })

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------
    def calculate_metrics(self):
        total = len(self.results)
        correct = sum(r["correct"] for r in self.results)

        self.metrics["total_tests"] = total
        self.metrics["accuracy"] = correct / total if total else 0.0
        self.metrics["error_rate"] = 1 - self.metrics["accuracy"]

        self.metrics["avg_confidence"] = statistics.mean(
            r["confidence"] for r in self.results
        )

        self.metrics["avg_latency_ms"] = statistics.mean(
            r["latency_ms"] for r in self.results
        )

        # Per-intent accuracy (by expected intent)
        per_intent = defaultdict(list)
        for r in self.results:
            per_intent[r["expected"]].append(r["correct"])

        self.metrics["per_intent_accuracy"] = {
            intent: sum(vals) / len(vals)
            for intent, vals in per_intent.items()
        }

        # Ambiguity detection accuracy
        amb_cases = [r for r in self.results if r["ambiguous_expected"]]
        if amb_cases:
            self.metrics["ambiguity_accuracy"] = (
                sum(r["ambiguous_predicted"] for r in amb_cases)
                / len(amb_cases)
            )
        else:
            self.metrics["ambiguity_accuracy"] = None

        # Match quality distribution
        quality_dist = defaultdict(int)
        for r in self.results:
            quality_dist[r["match_quality"]] += 1

        self.metrics["match_quality_distribution"] = dict(quality_dist)

    # -------------------------------------------------
    # Report
    # -------------------------------------------------
    def generate_report(self):
        print("=== LCN Test Results ===")
        print(f"Total tests: {self.metrics['total_tests']}")
        print(f"Accuracy: {self.metrics['accuracy'] * 100:.2f}%")
        print(f"Error rate: {self.metrics['error_rate'] * 100:.2f}%")
        print(f"Average confidence: {self.metrics['avg_confidence']:.3f}")
        print(f"Average latency: {self.metrics['avg_latency_ms']:.2f} ms")

        print("\nPer-intent accuracy:")
        for intent, acc in sorted(self.metrics["per_intent_accuracy"].items()):
            print(f"  {intent}: {acc * 100:.2f}%")

        if self.metrics["ambiguity_accuracy"] is not None:
            print(f"\nAmbiguity detection accuracy: "
                  f"{self.metrics['ambiguity_accuracy'] * 100:.2f}%")

        print("\nMatch quality distribution:")
        for k, v in self.metrics["match_quality_distribution"].items():
            print(f"  {k}: {v}")
