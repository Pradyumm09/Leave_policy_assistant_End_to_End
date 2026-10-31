def build_hr_guidance(question: str) -> dict:

    answer = """
This appears to be a request related to your individual leave
situation or the action required to request/approve leave.

The Leave Policy Assistant can explain the documented leave policy,
but it cannot submit, approve, or modify an employee's leave request.

For an urgent or immediate leave request, please contact your
reporting manager and/or HR and follow the organization's official
leave application process.

### Dummy HR Contact

Email: hr@example.com

### Suggested Email

Subject: Urgent Leave Request

Hi HR Team,

I would like to request leave due to an urgent personal reason.
Please let me know the required process and approval steps for my
leave request.

Regards,
[Employee Name]
[Employee ID]
"""

    return {
        "answer": answer.strip(),
        "sources": [],
        "confidence": 1.0,
        "llm_provider": None,
        "intent": "HR_ACTION",
    }
