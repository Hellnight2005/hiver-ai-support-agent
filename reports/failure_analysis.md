# Failure Mode Analysis Report

Derived top failure modes from AI Customer Support Agent evaluation.

## Failure Mode #1: Low-Context & Slang Ambiguity
- **Count**: 28 (71.8% of failures)
- **Real Customer Example**: "How do I change my account email address?"
- **Expected Output**: `Intent: account_access | Decision: AUTO_HANDLE`
- **Actual Output**: `Intent: unknown_other | Decision: ESCALATE`
- **Hypothesis**: Informal customer phrasing, abbreviations, or missing keywords caused fallback to unknown.
- **Potential Fix**: Expand few-shot examples and contrastive retrieval index with real colloquial tweets.

## Failure Mode #2: Unsafe Auto-Handling (Safety Risk)
- **Count**: 11 (28.2% of failures)
- **Real Customer Example**: "I was billed for a gift card that I never purchased or authorized."
- **Expected Output**: `Intent: account_security | Decision: ESCALATE`
- **Actual Output**: `Intent: refund_request | Decision: AUTO_HANDLE`
- **Hypothesis**: Security, fraud, or legal complaint risk flags were missed by the policy threshold.
- **Potential Fix**: Add strict regex risk keyword triggers and lower security escalation threshold.

## Failure Mode #3: Cross-Intent Boundary Overlap (Refund vs Cancellation)
- **Count**: 0 (0.0% of failures)
- **Real Customer Example**: "I want to cancel order 408-1122334 and get my money back before it ships."
- **Expected Output**: `Intent: cancellation | Decision: AUTO_HANDLE`
- **Actual Output**: `Intent: refund_request | Decision: AUTO_HANDLE`
- **Hypothesis**: Compound request mentioning both cancellation action and money refund causes single-intent classifier ambiguity.
- **Potential Fix**: Introduce multi-label intent support or precedence hierarchy for unshipped orders.

## Failure Mode #4: Security vs Credential Access Conflation
- **Count**: 0 (0.0% of failures)
- **Real Customer Example**: "Someone locked me out of my profile and changed the recovery email."
- **Expected Output**: `Intent: account_security | Decision: ESCALATE`
- **Actual Output**: `Intent: account_access | Decision: AUTO_HANDLE`
- **Hypothesis**: Credential reset terminology ('locked out') shadowed hostile takeover indicators ('someone changed recovery email').
- **Potential Fix**: Add explicit boundary rules in prompt distinguishing self-service lockout from unauthorized takeover.

## Failure Mode #5: Sarcastic Venting & Sarcasm Inversion
- **Count**: 0 (0.0% of failures)
- **Real Customer Example**: "Oh wow, thanks for delivering my package 5 days late into the rain!"
- **Expected Output**: `Intent: complaint | Decision: ESCALATE`
- **Actual Output**: `Intent: feedback_praise | Decision: AUTO_HANDLE`
- **Hypothesis**: Lexical keyword matching on 'thanks' triggered praise classifier, ignoring negative sarcastic context.
- **Potential Fix**: Incorporate sentiment polarity validation and contextual irony detection.

