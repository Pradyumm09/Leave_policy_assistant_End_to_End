import re
from app.guardrails.intent_classifier import classify_intent


LEAVE_POLICY_KEYWORDS = {
    "leave",
    "leaves",
    "annual",
    "casual",
    "sick",
    "maternity",
    "paternity",
    "adoption",
    "bereavement",
    "compensatory",
    "holiday",
    "holidays",
    "unpaid",
    "lwop",
    "absence",
    "notice period",
    "carry forward",
    "encashment",
    "vacation",
    "time off",
}


INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"ignore all previous instructions",
    r"system prompt",
    r"developer message",
    r"reveal your prompt",
    r"show your instructions",
    r"jailbreak",
]


def validate_question(question: str) -> tuple[bool, str]:
    """
    Validate whether the question is suitable for the
    Leave Policy Assistant.
    """

    question = question.strip()

    if not question:
        return False, "Please enter a question."

    if len(question) > 1000:
        return False, "Question is too long. Please keep it under 1000 characters."

    # Prompt injection detection
    lowered_question = question.lower()

    for pattern in INJECTION_PATTERNS:

        if re.search(pattern, lowered_question):
            return (
                False,
                "I can only answer questions related to the leave policy.",
            )

    # Topic restriction
    contains_leave_topic = any(
        keyword in lowered_question
        for keyword in LEAVE_POLICY_KEYWORDS
    )

    intent = classify_intent(question)
    if intent == "HR_ACTION":
        return True, "HR_ACTION"


    contains_leave_topic = any(
        keyword in lowered_question
    for keyword in LEAVE_POLICY_KEYWORDS
)

    if not contains_leave_topic:
        return (
            False,
            "I can only answer questions related "
            "to the organization's leave policy.",
        )
    return True, "POLICY"
