import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.agent import SupportAgent


def main():
    print("==================================================================")
    print(" HIVER AI SUPPORT AGENT - INTERACTIVE CLI DEMO")
    print(" Type 'exit' or 'quit' to end the session.")
    print("==================================================================\n")

    agent = SupportAgent()

    while True:
        try:
            user_input = input("\nCustomer: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting CLI session. Goodbye!")
                break

            res = agent.process(user_input)

            print(f"\nIntent:           {res.intent['name']}")
            print(f"Confidence:       {res.intent['confidence']:.2f}")
            print(f"Decision:         {res.decision}")
            print(f"Reason:           {res.decision_reason}")
            if res.risk_flags:
                print(f"Risk Flags:       {', '.join(res.risk_flags)}")
            
            print(f"\nDraft Reply:\n\"{res.reply}\"")

            if res.retrieval:
                print("\nEvidence:")
                for idx, ev in enumerate(res.retrieval[:2], 1):
                    print(f"  {idx}. [Sim: {ev.similarity:.2f}] \"{ev.customer_message}\" -> \"{ev.agent_response}\"")

        except KeyboardInterrupt:
            print("\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError processing query: {str(e)}")


if __name__ == "__main__":
    main()
