import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

logger = logging.getLogger('tess.tests.test_binary_classifier')

@dataclass
class TestCase:
    text: str
    expected_intent: str
    id: str = ""
    
@dataclass
class TestResult:
    test_id: str
    text: str
    expected_intent: str
    predicted_intent: str
    confidence: float
    latency_ms: float
    correct: bool
    note: str = ""

class BinaryClassifierTester:
    def __init__(self, classifier, test_cases_path: str):
        """
        Initialize the tester with a classifier instance and path to test cases JSON.
        
        Args:
            classifier: Instance of Classifier class
            test_cases_path: Path to JSON file containing test cases
        """
        self.classifier = classifier
        self.test_cases = self._load_test_cases(test_cases_path)
        self.results: List[TestResult] = []
        
    def _load_test_cases(self, path: str) -> List[TestCase]:
        """Load test cases from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        test_cases = []
        for idx, case in enumerate(data.get('test_cases', [])):
            test_cases.append(TestCase(
                text=case['text'],
                expected_intent=case['expected_intent'],
                id=case.get('id', f"test_{idx}")
            ))
        
        logger.info(f"Loaded {len(test_cases)} test cases from {path}")
        return test_cases
    
    def run_tests(self, threshold: float = 0.65) -> Dict[str, Any]:
        """
        Run all test cases and collect results.
        
        Args:
            threshold: Confidence threshold for command classification
            
        Returns:
            Dictionary containing metrics and detailed results
        """
        logger.info(f"Running {len(self.test_cases)} tests with threshold={threshold}")
        self.results = []
        
        for test_case in self.test_cases:
            start_time = time.perf_counter()
            prediction = self.classifier.predict(test_case.text, threshold=threshold)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            
            result = TestResult(
                test_id=test_case.id,
                text=test_case.text,
                expected_intent=test_case.expected_intent,
                predicted_intent=prediction['intent'],
                confidence=prediction['confidence'],
                latency_ms=latency_ms,
                correct=prediction['intent'] == test_case.expected_intent,
                note=prediction.get('note', '')
            )
            
            self.results.append(result)
        
        metrics = self._calculate_metrics()
        logger.info(f"Tests completed. Overall accuracy: {metrics['overall_accuracy']:.2%}")
        
        return {
            'metrics': metrics,
            'results': [asdict(r) for r in self.results]
        }
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate accuracy and latency metrics."""
        if not self.results:
            return {}
        
        # Overall metrics
        total = len(self.results)
        correct = sum(1 for r in self.results if r.correct)
        overall_accuracy = correct / total if total > 0 else 0
        
        # Latency metrics
        latencies = [r.latency_ms for r in self.results]
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        # Per-intent metrics
        intent_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        for result in self.results:
            intent = result.expected_intent
            intent_stats[intent]['total'] += 1
            if result.correct:
                intent_stats[intent]['correct'] += 1
        
        per_intent_accuracy = {}
        for intent, stats in intent_stats.items():
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            per_intent_accuracy[intent] = {
                'accuracy': accuracy,
                'correct': stats['correct'],
                'total': stats['total']
            }
        
        return {
            'overall_accuracy': overall_accuracy,
            'total_tests': total,
            'correct_predictions': correct,
            'incorrect_predictions': total - correct,
            'latency': {
                'average_ms': avg_latency,
                'min_ms': min_latency,
                'max_ms': max_latency
            },
            'per_intent_accuracy': per_intent_accuracy
        }
    
    def print_summary(self):
        """Print a formatted summary of test results."""
        if not self.results:
            print("No test results available. Run tests first.")
            return
        
        metrics = self._calculate_metrics()
        
        print("\n" + "="*70)
        print("BINARY CLASSIFIER TEST SUMMARY")
        print("="*70)
        
        print(f"\n📊 Overall Performance:")
        print(f"   Accuracy: {metrics['overall_accuracy']:.2%} "
              f"({metrics['correct_predictions']}/{metrics['total_tests']})")
        
        print(f"\n⚡ Latency:")
        print(f"   Average: {metrics['latency']['average_ms']:.2f} ms")
        print(f"   Min: {metrics['latency']['min_ms']:.2f} ms")
        print(f"   Max: {metrics['latency']['max_ms']:.2f} ms")
        
        print(f"\n🎯 Per-Intent Accuracy:")
        for intent, stats in metrics['per_intent_accuracy'].items():
            print(f"   {intent.upper()}: {stats['accuracy']:.2%} "
                  f"({stats['correct']}/{stats['total']})")
        
        # Show failed cases
        failed = [r for r in self.results if not r.correct]
        if failed:
            print(f"\n❌ Failed Cases ({len(failed)}):")
            for result in failed[:10]:  # Show first 10 failures
                print(f"   • \"{result.text}\"")
                print(f"     Expected: {result.expected_intent}, "
                      f"Got: {result.predicted_intent} "
                      f"(confidence: {result.confidence:.2%})")
                if result.note:
                    print(f"     Note: {result.note}")
        
        print("\n" + "="*70 + "\n")
    
    def save_results(self, output_path: str):
        """Save detailed results to JSON file."""
        metrics = self._calculate_metrics()
        output_data = {
            'metrics': metrics,
            'results': [asdict(r) for r in self.results]
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Import your classifier
    from core.binary_classifier import Classifier
    
    # Initialize classifier and tester
    classifier = Classifier()
    tester = BinaryClassifierTester(
        classifier=classifier,
        test_cases_path='tests/data/binary_classifier_test_cases.json'
    )
    
    # Run tests
    results = tester.run_tests(threshold=0.65)
    
    # Print summary
    tester.print_summary()
    
    # Save detailed results
    tester.save_results('tests/results/binary_classifier_results.json')
