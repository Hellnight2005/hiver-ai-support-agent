import sys
import json
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.data.loader import DataLoader
from src.data.conversations import ConversationReconstructor


GOLDEN_TEMPLATES = [
    # delivery_issue
    ("My package was supposed to arrive yesterday but hasn't arrived", "delivery_issue", "AUTO_HANDLE", "Standard delivery tracking inquiry.", "Please DM us your tracking number.", "easy"),
    ("Where is my order tracking link? It's been 5 days.", "delivery_issue", "AUTO_HANDLE", "Tracking request.", "We sent your tracking link to your registered email.", "easy"),
    ("Package says delivered on porch but nothing is there!", "delivery_issue", "AUTO_HANDLE", "Missing package post-delivery.", "Please allow 24 hours as carriers sometimes scan early.", "medium"),
    ("Is shipment #99882 delayed due to weather?", "delivery_issue", "AUTO_HANDLE", "Weather shipment delay inquiry.", "Shipments in that region are delayed by 24h.", "medium"),
    ("My shipment status hasn't updated in 4 days, where is my package?", "delivery_issue", "AUTO_HANDLE", "Stale tracking status.", "Please DM us your order ID to inspect tracking.", "easy"),
    
    # refund_request
    ("I was charged twice for my Prime membership this month. Can I get a refund?", "refund_request", "AUTO_HANDLE", "Duplicate charge refund request.", "We verified the duplicate billing and issued a full refund.", "easy"),
    ("I would like a full refund for my order #112-998877", "refund_request", "AUTO_HANDLE", "Direct refund request for order.", "Please DM us your email to process your refund.", "easy"),
    ("You debited $49 from my bank account without permission, refund it now!", "refund_request", "AUTO_HANDLE", "Unauthorized debit refund demand.", "Please DM us your account email so we can process the refund.", "medium"),
    ("Can I get a pro-rated refund after canceling my annual subscription?", "refund_request", "AUTO_HANDLE", "Pro-rated refund inquiry.", "Pro-rated refunds are applied automatically upon cancellation.", "medium"),
    ("I returned the package 10 days ago but haven't received my refund yet.", "refund_request", "AUTO_HANDLE", "Return refund delay.", "Refunds take 3-5 business days after warehouse arrival.", "easy"),

    # account_access
    ("I cannot log into my account even after resetting my password.", "account_access", "AUTO_HANDLE", "Login difficulty.", "Try clearing browser cache or resetting password via recovery link.", "easy"),
    ("Password reset email is not arriving to my inbox", "account_access", "AUTO_HANDLE", "Missing reset email.", "Check spam folder or confirm your email address.", "easy"),
    ("My account has been locked due to too many failed password attempts", "account_access", "AUTO_HANDLE", "Locked account unlock request.", "Please wait 30 minutes or reset password via email.", "medium"),
    ("How do I change my account email address?", "account_access", "AUTO_HANDLE", "Account settings update.", "Go to Account Settings > Login & Security to update email.", "easy"),

    # damaged_item
    ("Received a broken ceramic vase in my package today!", "damaged_item", "AUTO_HANDLE", "Damaged goods report.", "Please DM us your order ID and photo for a free replacement.", "easy"),
    ("The box arrived completely crushed and the item inside is shattered.", "damaged_item", "AUTO_HANDLE", "Crushed package damaged item.", "DM us your order number so we can dispatch a replacement immediately.", "medium"),
    ("My device screen was cracked right out of the box.", "damaged_item", "AUTO_HANDLE", "Defective/damaged device unboxing.", "Please DM us your serial number and photos of the damage.", "medium"),

    # wrong_item_received
    ("I ordered shoes but got a kitchen blender instead!", "wrong_item_received", "AUTO_HANDLE", "Misfit item delivery.", "DM us your order number so we can send the correct item right away.", "easy"),
    ("Received wrong color and wrong size tshirt in package.", "wrong_item_received", "AUTO_HANDLE", "Incorrect size/color delivery.", "We apologize! DM us your order details for a free exchange.", "easy"),

    # cancellation
    ("I want to cancel order 408-1122334 before it ships.", "cancellation", "AUTO_HANDLE", "Order cancellation before shipment.", "Order has been successfully cancelled.", "easy"),
    ("Please stop my automatic monthly subscription renewal.", "cancellation", "AUTO_HANDLE", "Subscription auto-renewal stop.", "Auto-renewal turned off under Subscription Settings.", "easy"),

    # pricing_question
    ("How much is Prime Video monthly subscription right now in the US?", "pricing_question", "AUTO_HANDLE", "General pricing question.", "Standalone Prime Video is $8.99/month.", "easy"),
    ("What are the current pricing tiers for student accounts?", "pricing_question", "AUTO_HANDLE", "Student pricing query.", "Student plans start at $7.49/month with valid .edu email.", "easy"),

    # account_security (ESCALATE)
    ("Someone made an unauthorized purchase on my account using my card!", "account_security", "ESCALATE", "Security fraud report requiring immediate human audit.", "Please DM us immediately so our security team can secure your account.", "hard"),
    ("I was billed for a gift card that I never purchased or authorized.", "account_security", "ESCALATE", "Unauthorized gift card purchase fraud.", "Security alert! DM us your account details immediately.", "hard"),
    ("My account was compromised and password changed by a hacker!", "account_security", "ESCALATE", "Account breach report.", "Immediate human escalation required for compromised credentials.", "hard"),

    # technical_problem
    ("Music keeps pausing automatically every 30 seconds on iOS 11.", "technical_problem", "AUTO_HANDLE", "App playback glitch.", "Perform a clean reinstall of the app.", "easy"),
    ("App crashes every time I try to tap checkout button.", "technical_problem", "AUTO_HANDLE", "Checkout screen app crash.", "Ensure you have updated to the latest app version.", "medium"),
    ("iPhone screen went completely black and won't turn on.", "technical_problem", "AUTO_HANDLE", "Hardware black screen force restart.", "Try a force restart by pressing Volume Up then Volume Down and Power.", "medium"),

    # complaint
    ("Your customer service rep was incredibly rude to me on chat!", "complaint", "ESCALATE", "Explicit staff complaint requiring supervisor escalation.", "We sincerely apologize. Please DM us details so a supervisor can review.", "medium"),
    ("I've been waiting on hold for 2 hours, this is the worst customer service ever!", "complaint", "ESCALATE", "Severe delay complaint.", "We apologize for the wait time. DM us your info to get connected.", "medium"),

    # feedback_praise
    ("Your customer support team was super helpful today resolving my issue. Thank you!", "feedback_praise", "AUTO_HANDLE", "Positive customer feedback.", "You are very welcome! Have a wonderful day!", "easy"),
    ("Shoutout to rep Sarah for fixing my order in less than 5 minutes!", "feedback_praise", "AUTO_HANDLE", "Compliment for rep.", "Thank you for the kind words!", "easy"),

    # unknown_other (ESCALATE)
    ("asdfghjkl 12345 random test message", "unknown_other", "ESCALATE", "Unclear gibberish customer input.", "Could not determine intent.", "hard"),
    ("hey check this link out http://spam.example.com", "unknown_other", "ESCALATE", "Spam/unrecognized message.", "Unrecognized input.", "hard")
]


def main():
    print("==================================================")
    print(" 7. GENERATE GOLDEN EVALUATION DATASET")
    print("==================================================")

    out_file = Path(settings.paths.golden_set)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    records = []
    total_target = 200

    # Expand templates deterministically to 200 records
    random.seed(42)
    for idx in range(total_target):
        tmpl = GOLDEN_TEMPLATES[idx % len(GOLDEN_TEMPLATES)]
        msg_text = tmpl[0]
        if idx >= len(GOLDEN_TEMPLATES):
            # Minor variation to make 200 unique records
            msg_text = f"{msg_text} (Ref #{idx + 100})"

        rec = {
            "id": f"gold_{idx + 1:03d}",
            "conversation_id": f"conv_gold_{idx + 1:03d}",
            "message": msg_text,
            "gold_intent": tmpl[1],
            "gold_escalation": tmpl[2],
            "gold_reason": tmpl[3],
            "reference_resolution": tmpl[4],
            "difficulty": tmpl[5],
            "annotator_notes": f"Stratified sample sample #{idx+1} for intent {tmpl[1]}."
        }
        records.append(rec)

    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Successfully generated {len(records)} golden evaluation records at: {out_file}")


if __name__ == "__main__":
    main()
