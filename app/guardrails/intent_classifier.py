HR_ACTION_KEYWORDS = {
    "apply",
    "application",
    "request",
    "approval",
    "approve",
    "manager",
    "hr",
    "human resources",
    "contact hr",
    "contact manager",
    "urgent leave",
    "immediate leave",
    "leave tomorrow",
    "leave today",
    "take leave tomorrow",
    "take leave today",
    "emergency leave",
    "how do i apply",
    "how can i apply",
    "submit leave",
    "submit my leave",
    "cancel my leave",
    "change my leave",
    "modify my leave",
}


def classify_intent(question: str) -> str:

    question = question.lower().strip()

    # -----------------------------------------
    # HR / Personal action request
    # -----------------------------------------

    for keyword in HR_ACTION_KEYWORDS:

        if keyword in question:
            return "HR_ACTION"

    # -----------------------------------------
    # Leave-policy question
    # -----------------------------------------

    return "POLICY"
