# Golden Set Annotation Guidelines

## Overview
This document provides rules and standards for annotating customer support evaluation examples in `data/golden/golden_set.jsonl`.

## Golden Record Schema
Each line in `golden_set.jsonl` is a JSON object formatted as follows:

```json
{
  "id": "gold_001",
  "conversation_id": "conv_root_123",
  "message": "My order 112-9847291-098234 has not arrived yet and it was supposed to be delivered yesterday.",
  "gold_intent": "delivery_issue",
  "gold_escalation": "AUTO_HANDLE",
  "gold_reason": "Standard delayed delivery query with order number provided.",
  "reference_resolution": "We are sorry to hear that! Please send us a DM with your tracking number so we can check the status.",
  "difficulty": "easy",
  "annotator_notes": "Clear intent and safe for auto-handling."
}
```

## Intent Taxonomy Definitions
1. `delivery_issue`: Package tracking, delayed shipment, or missing delivery.
2. `refund_request`: Requesting reimbursement or money back for charges.
3. `account_access`: Password reset, login issues, account lockout.
4. `damaged_item`: Broken or physically damaged goods upon arrival.
5. `wrong_item_received`: Received incorrect product/color/size.
6. `cancellation`: Requesting to cancel order or subscription before shipment.
7. `pricing_question`: Inquiries about costs, pricing plans, student discounts.
8. `account_security`: Suspected fraud, unauthorized charges, compromised account. **MUST ESCALATE**.
9. `technical_problem`: App crashes, video buffering errors, hardware issues.
10. `complaint`: Explicit dissatisfaction with agent or brand service.
11. `feedback_praise`: Positive feedback, compliments, or thank-you messages.
12. `unknown_other`: Ambiguous, truncated, or gibberish messages. **MUST ESCALATE**.

## Escalation Policy Rules
- **AUTO_HANDLE**: Standard issues where historical resolution guidance is clear, non-sensitive, and safe to automate without private account modifications.
- **ESCALATE**: 
  - Security threats, fraud, unauthorized billing (`account_security`).
  - Legal threats, lawyers, law enforcement mentions.
  - Severe explicit complaints.
  - Ambiguous/gibberish messages (`unknown_other`).
  - High-value account action requests requiring verified human identity check.

## Difficulty Categorization
- `easy`: Unambiguous single-intent message matching exact taxonomy definition.
- `medium`: Contains minor informal language, typos, or secondary context.
- `hard`: Multi-intent, sarcasm, boundary cases between similar intents (e.g. refund vs cancellation), or high ambiguity.
