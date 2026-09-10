from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.agent.schemas import AgentResponse, GroundingEvidence


class TFIDFBaseline:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
        self.classifier = LogisticRegression(max_iter=500)
        self.train_messages: List[str] = []
        self.train_responses: List[str] = []
        self.train_conv_ids: List[str] = []
        self.train_matrix = None
        self.is_fitted = False

    def fit(self, training_items: List[Dict[str, Any]]):
        if not training_items:
            return
        
        texts = [it["customer_message"] for it in training_items]
        intents = [it.get("intent", "unknown_other") for it in training_items]
        responses = [it["agent_response"] for it in training_items]
        conv_ids = [it["conversation_id"] for it in training_items]

        self.train_messages = texts
        self.train_responses = responses
        self.train_conv_ids = conv_ids

        self.train_matrix = self.vectorizer.fit_transform(texts)
        
        # Fit classifier if at least 2 distinct classes
        if len(set(intents)) >= 2:
            self.classifier.fit(self.train_matrix, intents)
            self.is_fitted = True

    def process(self, customer_message: str, exclude_conv_ids: Optional[List[str]] = None) -> AgentResponse:
        if not self.is_fitted or self.train_matrix is None:
            return AgentResponse(
                message=customer_message,
                intent={"name": "unknown_other", "confidence": 0.0, "reason": "TFIDF baseline not fitted"},
                retrieval=[],
                reply="Please DM us your details.",
                decision="ESCALATE",
                decision_reason="TFIDF baseline unfitted fallback",
                risk_flags=[]
            )

        q_vec = self.vectorizer.transform([customer_message])
        
        # Predict intent
        intent_probs = self.classifier.predict_proba(q_vec)[0]
        max_idx = np.argmax(intent_probs)
        pred_intent = self.classifier.classes_[max_idx]
        confidence = float(intent_probs[max_idx])

        # TF-IDF Cosine Retrieval
        scores = (self.train_matrix * q_vec.T).toarray().flatten()
        sorted_indices = np.argsort(scores)[::-1]
        
        exclude_set = set(exclude_conv_ids or [])
        evidence = []
        best_reply = "Please DM us your order details."

        for idx in sorted_indices:
            cid = self.train_conv_ids[idx]
            if cid in exclude_set:
                continue
            sim = float(scores[idx])
            best_reply = self.train_responses[idx]
            evidence.append(GroundingEvidence(
                conversation_id=cid,
                similarity=sim,
                customer_message=self.train_messages[idx],
                agent_response=best_reply
            ))
            if len(evidence) >= 5:
                break

        top_sim = evidence[0].similarity if evidence else 0.0
        is_sensitive = pred_intent in ["account_security", "complaint", "unknown_other"]
        decision = "AUTO_HANDLE" if (confidence >= 0.50 and top_sim >= 0.15 and not is_sensitive) else "ESCALATE"

        return AgentResponse(
            message=customer_message,
            intent={"name": pred_intent, "confidence": confidence, "reason": "TF-IDF LogisticRegression classification"},
            retrieval=evidence,
            reply=best_reply,
            decision=decision,
            decision_reason=f"TF-IDF decision based on conf={confidence:.2f}, sim={top_sim:.2f}",
            risk_flags=[]
        )
