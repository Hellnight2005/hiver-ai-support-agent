# Failure Mode Analysis Report

Derived top failure modes from AI Customer Support Agent evaluation.

## Failure Mode #1: Unnecessary Escalation Failure
- **Count**: 28 (71.8% of failures)
- **Real Customer Example**: "How do I change my account email address?"
- **Expected Output**: `Intent: account_access | Decision: AUTO_HANDLE`
- **Actual Output**: `Intent: unknown_other | Decision: ESCALATE`
- **Hypothesis**: Retrieval confidence fell just below threshold despite valid intent.
- **Potential Fix**: Tune retrieval similarity threshold on validation data.

## Failure Mode #2: Unsafe Auto-Handling Failure
- **Count**: 11 (28.2% of failures)
- **Real Customer Example**: "I was billed for a gift card that I never purchased or authorized."
- **Expected Output**: `Intent: account_security | Decision: ESCALATE`
- **Actual Output**: `Intent: refund_request | Decision: AUTO_HANDLE`
- **Hypothesis**: Security/fraud risk flags were missed by threshold policy.
- **Potential Fix**: Add strict keyword triggers and lower security escalation threshold.

