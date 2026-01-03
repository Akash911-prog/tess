import logging

from core.lcn import LCN
from core.context_manager import ContextManager, State
from core.stt import STT
from libs.logger_config import setup_logging


setup_logging(
    log_level="DEBUG",
    log_dir="logs",
    enable_console=True,
    enable_file=True
)

logger = logging.getLogger('tess.main')

class Main():

    def __init__(self) -> None:

        self.normalizer = LCN(model="MongoDB/mdbr-leaf-ir")
        self.context = ContextManager()
        self.stt = STT(state_manager=self.context)
        # self.indicator = self.stt.indicator

    def _compare_lcn_results(self, old_tester, new_tester):
        """
        Compare two LCNTester runs (old vs new).

        Returns a structured dict with improvements and regressions.
        """

        old_metrics = old_tester.metrics
        new_metrics = new_tester.metrics

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


    def run(self) -> None:
        logger.info("ready")
        text = ''
        while text.lower() != "close":
            logger.info("ready")
            text = self.stt.start_listening()
            self.context.set_state(State.PROCESSING)
            logger.info(text)
            logger.info("processing")
            result = self.normalizer.normalize(text)
            logger.info(result)
            self.context.set_state(State.IDLE)
            logger.info("idle")
        self.stt.close()

    


if __name__ == "__main__":
    main = Main()
    main.run()
