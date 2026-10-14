"""Transparent, deterministic clue checks; computer-vision results stay separate from case facts."""

from typing import Dict, List


MISSIONS = ["eyes-online", "missing-clue", "red-bag", "follow-trail"]


def validate_clue(mission: str, answer: str) -> Dict[str, object]:
    normalized = " ".join(answer.strip().lower().split())
    answers = {
        "eyes-online": {"23:42", "2342"},
        "missing-clue": {"phone and prototype missing; additional cup present", "phone prototype missing extra cup", "mobile phone and prototype missing; an additional cup present"},
        "red-bag": {"aarav", "aarav owns the backpack"},
        "follow-trail": {"kabir", "kabir carried the backpack", "23:44"},
    }
    if mission not in answers:
        return {"correct": False, "message": "That mission is not in this case file."}
    if mission == "missing-clue":
        correct = all(term in normalized for term in ("phone", "prototype", "cup")) and any(term in normalized for term in ("extra", "additional"))
    elif mission == "follow-trail":
        correct = "kabir" in normalized and ("23:44" in normalized or "carried" in normalized or "carrier" in normalized or "exit" in normalized)
    else:
        correct = normalized in answers[mission]
    hints = {
        "eyes-online": "Look at the enhanced clock crop; enter the displayed time as HH:MM.",
        "missing-clue": "Compare the 23:42 and 23:43 scenes. The detector is optional; report the scene comparison explicitly.",
        "red-bag": "The case file identifies the owner. A color mask cannot establish ownership.",
        "follow-trail": "Use the observed trail for movement, then combine it with independent case-file evidence for the carrier's name.",
    }
    return {"correct": correct, "message": "Clue accepted. Evidence added to the board." if correct else hints[mission]}


def validate_accusation(answers: Dict[str, str], evidence_ids: List[str]) -> Dict[str, object]:
    expected = {"culprit": "kabir", "owner": "aarav", "carrier": "kabir", "incident": "23:42", "corruption": "23:44"}
    normalized = {key: " ".join(str(value).strip().lower().split()) for key, value in answers.items()}
    wrong = [key for key, value in expected.items() if normalized.get(key) != value]
    required = {"ev-time", "ev-scene", "ev-owner", "ev-trail", "ev-log"}
    missing_evidence = sorted(required.difference(evidence_ids))
    if wrong:
        hints = {"culprit": "Do not stop at bag ownership. Combine movement evidence with the independent system log.", "owner": "The case file, not the color mask, establishes ownership.", "carrier": "Who does the movement clue place on the corridor route?", "incident": "Inspect the enhanced clock crop.", "corruption": "The prepared fictional system log is separate from image processing."}
        return {"correct": False, "message": hints[wrong[0]], "incorrect_fields": wrong, "missing_evidence": missing_evidence}
    if missing_evidence:
        return {"correct": False, "message": "Select the five independent evidence items on the board before submitting.", "incorrect_fields": [], "missing_evidence": missing_evidence}
    return {"correct": True, "message": "Case solved. The conclusion is supported by independent visual and case-file evidence.", "incorrect_fields": [], "missing_evidence": []}