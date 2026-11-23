"""Runs one structured agent step as a real CrewAI agent (owner: An).

The graph decides who works next; inside a node, ``agent_complete`` hands the step to the
CrewAI agent of ``agents/specs.py`` (role, goal, backstory, its own read-only tools) and
returns the same ``(answer, usage)`` pair as ``ModelClient.complete``, so a node can swap
one for the other.

Every model turn of the agent goes through the project's ``ModelClient`` (``ModelClientLLM``):
the shared budget, the spending cap and the usage log see each call, and CrewAI never
calls a provider by itself. ``BudgetExceeded`` always propagates.

If CrewAI is missing or the crew fails for another reason, the step falls back to one direct
call of the same client, so a measurement never stops because of the agent runtime. The
returned route says which path answered, for the event log.
"""

import os
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel

from ai_share_of_voice.agents.specs import AGENT_SPECS
from ai_share_of_voice.exceptions import BudgetExceeded
from ai_share_of_voice.schemas.llm import ModelClient, Usage

# CrewAI sends anonymous telemetry by default; a measurement must not call anyone else.
for _name in ("CREWAI_DISABLE_TELEMETRY", "OTEL_SDK_DISABLED"):
    os.environ.setdefault(_name, "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

Route = Literal["crewai", "direct"]
DISABLE_ENV = "AISOV_CREWAI"  # set to "off" to always use the direct call
CONTEXT_WINDOW = 131_072


def _prompt_of(messages: str | Sequence[Mapping[str, Any]]) -> str:
    if isinstance(messages, str):
        return messages
    blocks = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            content = "\n".join(
                str(part.get("text", "")) for part in content if isinstance(part, Mapping)
            )
        if content:
            blocks.append(f"[{message.get('role', 'user')}]\n{content}")
    return "\n\n".join(blocks)


def _llm_class() -> type:
    from crewai import BaseLLM

    class ModelClientLLM(BaseLLM):
        """A CrewAI LLM whose every turn is a call of the project's ``ModelClient``."""

        def __init__(self, client: ModelClient) -> None:
            super().__init__(model=str(getattr(client, "model", "model-client")), temperature=0)
            self._client = client
            self.usages: list[Usage] = []
            # CrewAI retries a failed turn by itself; after the first failure no further call
            # may reach the provider, or a full budget or a provider error would cost more.
            self.error: Exception | None = None

        def call(
            self,
            messages: Any,
            tools: Any = None,
            callbacks: Any = None,
            available_functions: Any = None,
            from_task: Any = None,
            from_agent: Any = None,
            response_model: type[BaseModel] | None = None,
            **kwargs: Any,
        ) -> str:
            if self.error is not None:
                raise self.error
            try:
                return self._turn(_prompt_of(messages), response_model)
            except Exception as error:
                self.error = error
                raise

        def _turn(self, prompt: str, response_model: type[BaseModel] | None) -> str:
            if response_model is not None:
                answer, usage = self._client.complete(prompt, response_model)
                self.usages.append(usage)
                return answer.model_dump_json()
            text_answer = getattr(self._client, "answer", None)
            if text_answer is None:
                raise RuntimeError("this client cannot answer in free text")
            text, usage = text_answer(prompt)
            self.usages.append(usage)
            return text

        def supports_function_calling(self) -> bool:
            return False  # CrewAI then uses its text format and runs the tools itself

        def supports_stop_words(self) -> bool:
            return False

        def get_context_window_size(self) -> int:
            return CONTEXT_WINDOW

    return ModelClientLLM


def _read_only_tool(name: str, text: str) -> Any:
    from crewai.tools import BaseTool

    class ReadOnlyTool(BaseTool):
        def _run(self, **_arguments: Any) -> str:
            return text

    return ReadOnlyTool(
        name=name,
        description=f"Read-only: returns the {name.replace('_', ' ')} of this run as text.",
    )


def _total(usages: Sequence[Usage], model: str) -> Usage:
    return Usage(
        model=usages[-1].model if usages else model,
        input_tokens=sum(item.input_tokens for item in usages),
        output_tokens=sum(item.output_tokens for item in usages),
        cost_usd=sum(item.cost_usd for item in usages),
    )


def _run_crew[T: BaseModel](
    agent_key: str,
    client: ModelClient,
    prompt: str,
    schema: type[T],
    tools: Mapping[str, str],
) -> tuple[T, Usage]:
    from crewai import Crew, Process, Task

    from ai_share_of_voice.agents.factory import build_agent

    spec = AGENT_SPECS[agent_key]
    llm = _llm_class()(client)
    agent = build_agent(
        spec,
        llm=llm,
        tools={name: _read_only_tool(name, tools.get(name, "")) for name in spec.tools},
    )
    task = Task(
        description=prompt.replace("{", "{{").replace("}", "}}"),
        expected_output=f"An answer that validates against the {schema.__name__} schema.",
        agent=agent,
        output_pydantic=schema,
    )
    try:
        result = Crew(
            agents=[agent], tasks=[task], process=Process.sequential, verbose=False
        ).kickoff()
    except Exception:
        if llm.error is not None:
            raise llm.error from None
        raise
    if llm.error is not None:  # CrewAI swallowed a failed turn: the answer cannot be trusted
        raise llm.error
    answer = result.pydantic
    if not isinstance(answer, schema):
        answer = schema.model_validate_json(result.raw)
    return answer, _total(llm.usages, llm.model)


def agent_complete[T: BaseModel](
    agent_key: str,
    client: ModelClient,
    prompt: str,
    schema: type[T],
    *,
    tools: Mapping[str, str] | None = None,
) -> tuple[T, Usage, Route]:
    """Run one step as the CrewAI agent ``agent_key``; fall back to a direct call.

    ``tools`` maps the tool names of the agent's spec to the read-only text they return.
    Raises what ``client.complete`` raises when the fallback fails, and ``BudgetExceeded``
    from either path.
    """
    if os.environ.get(DISABLE_ENV, "").strip().lower() != "off":
        try:
            answer, usage = _run_crew(agent_key, client, prompt, schema, tools or {})
        except BudgetExceeded:
            raise
        except Exception:  # noqa: BLE001 - any agent-runtime failure falls back to one call
            pass
        else:
            return answer, usage, "crewai"
    answer, usage = client.complete(prompt, schema)
    return answer, usage, "direct"
