import pytest
from pydantic import BaseModel

from ai_share_of_voice.agents.crew import DISABLE_ENV, agent_complete
from ai_share_of_voice.exceptions import BudgetExceeded
from ai_share_of_voice.llm_client import Budget, InvalidModelResponse, SimulatedClient
from ai_share_of_voice.writing import DraftReport


class Verdict(BaseModel):
    ok: bool


def client_with(respond, budget=None):
    calls = []

    def recorded(prompt):
        calls.append(prompt)
        return respond(prompt)

    client = SimulatedClient(budget or Budget(1.0), responders={Verdict: recorded})
    return client, calls


def test_the_step_runs_as_the_crewai_agent_of_its_spec():
    client, calls = client_with(lambda prompt: Verdict(ok=True))

    answer, usage, route = agent_complete("skeptic", client, "Is the draft fine?", Verdict)

    assert answer == Verdict(ok=True)
    assert route == "crewai"
    assert len(calls) == 1
    # The agent's role and the task reach the model through the project client.
    assert "Skeptical Reviewer" in calls[0] and "Is the draft fine?" in calls[0]
    assert usage.cost_usd == pytest.approx(client.budget.spent_usd)


def test_braces_in_the_prompt_are_not_read_as_crewai_placeholders():
    client, calls = client_with(lambda prompt: Verdict(ok=True))

    agent_complete("skeptic", client, 'Check {"brand": "Glovo"} please', Verdict)

    assert '{"brand": "Glovo"}' in calls[0]


def test_an_exhausted_budget_propagates_after_one_call_only():
    client, calls = client_with(lambda prompt: Verdict(ok=True), budget=Budget(0.0001))

    with pytest.raises(BudgetExceeded):
        agent_complete("writer", client, "Write it", Verdict)

    assert calls == []  # the reservation failed before any call, and nothing was retried


def test_an_invalid_answer_falls_back_to_one_direct_call():
    answers = iter([InvalidModelResponse("bad json"), Verdict(ok=False)])

    def respond(prompt):
        item = next(answers)
        if isinstance(item, Exception):
            raise item
        return item

    client, calls = client_with(respond)

    answer, _usage, route = agent_complete("skeptic", client, "Review", Verdict)

    assert (answer, route) == (Verdict(ok=False), "direct")
    assert len(calls) == 2  # one crew turn (not retried by CrewAI) and one direct call


def test_the_runtime_can_be_switched_off(monkeypatch):
    monkeypatch.setenv(DISABLE_ENV, "off")
    client, calls = client_with(lambda prompt: Verdict(ok=True))

    _answer, _usage, route = agent_complete("skeptic", client, "Review", Verdict)

    assert route == "direct"
    assert calls == ["Review"]


def test_the_writer_gets_its_read_only_tools():
    seen = []

    def respond(prompt):
        seen.append(prompt)
        return DraftReport(markdown="# Informe\n\nTexto suficiente.")

    client = SimulatedClient(Budget(1.0), responders={DraftReport: respond})

    _answer, _usage, route = agent_complete(
        "writer",
        client,
        "Rewrite the draft",
        DraftReport,
        tools={"report_template": "# T", "read_results": "{}"},
    )

    assert route == "crewai"
    assert "report_template" in seen[0] and "read_results" in seen[0]
