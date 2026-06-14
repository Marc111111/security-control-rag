from __future__ import annotations

from app.schemas import QueryPlan, SubQuestion


class RiskQuestionPlanner:
    def plan(self, question: str) -> QueryPlan:
        clean = question.strip()
        if not clean:
            raise ValueError("question cannot be empty")
        sub_questions = [
            SubQuestion(
                label="exposure_gap",
                focus="gap",
                question=f"What exposure or control gap is described? {clean}",
            ),
            SubQuestion(
                label="threats",
                focus="threat",
                question=f"What cybersecurity threats are relevant to this gap? {clean}",
            ),
            SubQuestion(
                label="vulnerabilities",
                focus="vulnerability",
                question=f"What vulnerabilities are exploited or created? {clean}",
            ),
            SubQuestion(
                label="risks",
                focus="risk",
                question=f"What business or compliance risks result? {clean}",
            ),
            SubQuestion(
                label="controls",
                focus="control",
                question=f"What controls mitigate the gap, vulnerabilities, or risks? {clean}",
            ),
            SubQuestion(
                label="frameworks",
                focus="compliance",
                question=f"What standards or framework references support the controls? {clean}",
            ),
        ]
        retrieval_queries = [item.question for item in sub_questions]
        return QueryPlan(
            original_question=clean,
            sub_questions=sub_questions,
            retrieval_queries=retrieval_queries,
        )
