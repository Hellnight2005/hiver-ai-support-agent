# Failure Mode Analysis Report

Derived top failure modes from AI Customer Support Agent evaluation.

## Failure Mode #1: Unnecessary Escalation Failure
- **Count**: 163 (93.7% of failures)
- **Real Customer Example**: "My package was supposed to arrive yesterday but hasn't arrived"
- **Expected Output**: `Intent: delivery_issue | Decision: AUTO_HANDLE`
- **Actual Output**: `Intent: delivery_issue | Decision: ESCALATE`
- **Hypothesis**: Retrieval confidence fell just below threshold despite valid intent.
- **Potential Fix**: Tune retrieval similarity threshold on validation data.

## Failure Mode #2: General Misclassification
- **Count**: 11 (6.3% of failures)
- **Real Customer Example**: "I was billed for a gift card that I never purchased or authorized."
- **Expected Output**: `Intent: account_security | Decision: ESCALATE`
- **Actual Output**: `Intent: refund_request | Decision: ESCALATE`
- **Hypothesis**: Model misidentified boundary features.
- **Potential Fix**: Refine prompt guidelines and contrastive examples.

