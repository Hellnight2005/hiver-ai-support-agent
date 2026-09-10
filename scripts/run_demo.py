import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.agent import SupportAgent


DEMO_MESSAGES = [
    "My order 112-9847291-098234 has not arrived yet and it was supposed to be delivered yesterday.",
    "I was charged twice for my Prime membership this month. Can I get a refund for the extra charge?",
    "Someone made an unauthorized purchase on my account using my stored credit card!",
    "Music keeps pausing automatically every 30 seconds on iOS 11.",
    "asdfghjkl random unknown test string"
]


def main():
    print("==================================================================")
    print(" HIVER AI SUPPORT AGENT - INSTANT DEMONSTRATION MODE")
    print("==================================================================\n")

    agent = SupportAgent()

    for idx, msg in enumerate(DEMO_MESSAGES, 1):
        print(f"--- [Query #{idx}] --------------------------------------------------")
        print(f"Customer Message: \"{msg}\"\n")

        res = agent.process(msg)

        print(f"Predicted Intent:  {res.intent['name']} (Confidence: {res.intent['confidence']:.2f})")
        print(f"Intent Diagnostic: {res.intent['reason']}")
        print(f"Policy Decision:   [{res.decision}]")
        print(f"Decision Reason:   {res.decision_reason}")
        if res.risk_flags:
            print(f"Risk Flags:        {', '.join(res.risk_flags)}")
        
        print("\nDraft Agent Reply:")
        print(f"  \"{res.reply}\"")

        if res.retrieval:
            top_ev = res.retrieval[0]
            print(f"\nTop Grounding Evidence (Similarity: {top_ev.similarity:.2f}):")
            print(f"  Matched Query: \"{top_ev.customer_message}\"")
            print(f"  Hist Response: \"{top_ev.agent_response}\"")

        print("------------------------------------------------------------------\n")

    print("Demo completed successfully!")


if __name__ == "__main__":
    main()
