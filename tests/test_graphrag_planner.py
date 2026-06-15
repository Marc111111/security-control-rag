from app.planning.planner import RiskQuestionPlanner


def test_risk_planner_decomposes_grc_question() -> None:
    plan = RiskQuestionPlanner().plan("No anti-malware is deployed. What should I document?")

    labels = [item.label for item in plan.sub_questions]

    assert labels == [
        "exposure_gap",
        "threats",
        "vulnerabilities",
        "risks",
        "controls",
        "frameworks",
    ]
    assert len(plan.retrieval_queries) == 6

