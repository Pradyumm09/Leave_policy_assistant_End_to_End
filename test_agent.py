from app.agents.policy_agent import answer_policy_question


if __name__ == "__main__":

    query = "How many days of paternity leave are allowed?"

    print("\n" + "=" * 60)
    print("CREWAI POLICY AGENT TEST")
    print("=" * 60)

    answer = answer_policy_question(query)

    print("\nFINAL ANSWER:")
    print(answer)

    print("=" * 60)