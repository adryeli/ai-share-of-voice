"""The chat API end to end in simulated mode: brief, run, approval, answers, safe downloads."""

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ai_share_of_voice.api.server import create_app
from ai_share_of_voice.config import load_settings
from ai_share_of_voice.ui.access import AccessGate
from ai_share_of_voice.ui.run_controller import RunController

BRIEF = {
    "brand": "Glovo",
    "competitors": ["Just Eat", "Uber Eats"],
    "sector": "food_delivery",
    "city": "Madrid",
    "country": "España",
    "plan": "quick",
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MODE", "simulated")
    monkeypatch.setenv("SEARCH_PROVIDER", "none")
    settings = load_settings()
    controller = RunController(settings, runs_dir=tmp_path, data_dir=Path("data"))
    return TestClient(create_app(settings, controller))


def wait(client, run_id, phases):
    for _ in range(200):
        view = client.get(f"/api/runs/{run_id}").json()
        if view["phase"] not in phases:
            return view
        time.sleep(0.1)
    raise AssertionError(f"run stuck in {view['phase']}")


def test_meta_lists_plans_and_stop_rules(client):
    meta = client.get("/api/meta").json()
    assert meta["mode"] == "simulated"
    assert [p["answers"] for p in meta["plans"].values()] == [10, 30, 100]
    assert meta["stop_rules"]


def test_parse_reads_the_sentence(client):
    parsed = client.post(
        "/api/brief/parse",
        json={"text": "Glovo frente a Just Eat y Uber Eats en Madrid, comida a domicilio"},
    ).json()
    assert parsed["brand"] == "Glovo" and parsed["sector_key"] == "food_delivery"


def test_a_question_naming_a_brand_is_refused(client):
    resp = client.post(
        "/api/runs",
        json={"brief": BRIEF, "questions": ["¿Es Glovo la mejor?", "¿Qué app uso?", "¿Cuál pido?"]},
    )
    assert resp.status_code == 422


def test_full_run(client):
    questions = client.post("/api/brief/questions", json=BRIEF).json()["questions"]
    run_id = client.post("/api/runs", json={"brief": BRIEF, "questions": questions}).json()[
        "run_id"
    ]
    view = wait(client, run_id, {"running"})
    assert view["phase"] == "waiting_approval"
    assert view["funnel"] and view["stop"]["reason_text"]
    assert all("/" not in (e["detail"] or "") for e in view["events"])
    assert client.post(f"/api/runs/{run_id}/decision", json={"approved": True}).status_code == 200
    wait(client, run_id, {"running", "waiting_approval"})
    results = client.get(f"/api/runs/{run_id}/results").json()
    assert results["headline"] and results["brands"]
    reply = client.post(f"/api/runs/{run_id}/ask", json={"question": "¿por qué paraste?"}).json()[
        "reply"
    ]
    assert reply == view["stop"]["reason_text"]
    assert client.get(f"/api/runs/{run_id}/files/client_report.md").status_code == 200
    assert client.get(f"/api/runs/{run_id}/files/..%2F..%2Fetc%2Fpasswd").status_code == 404
    assert client.get(f"/api/runs/{run_id}/files/settings.json").status_code == 404


def test_model_calls_need_the_password_when_the_gate_is_on(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MODE", "simulated")
    settings = load_settings()
    controller = RunController(settings, runs_dir=tmp_path, data_dir=Path("data"))
    gated = TestClient(create_app(settings, controller, gate=AccessGate("s3cret")))
    assert gated.post("/api/runs", json={"brief": BRIEF}).status_code in (401, 423)
    ok = gated.post("/api/access", headers={"X-Demo-Password": "s3cret"})
    assert ok.status_code == 200


@pytest.fixture
def gated(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MODE", "simulated")
    monkeypatch.setenv("SEARCH_PROVIDER", "none")
    settings = load_settings()
    controller = RunController(settings, runs_dir=tmp_path, data_dir=Path("data"))
    return TestClient(create_app(settings, controller, gate=AccessGate("s3cret")))


def test_measurements_need_the_password_when_the_gate_is_on(gated):
    password = {"X-Demo-Password": "s3cret"}
    questions = gated.post("/api/brief/questions", json=BRIEF).json()["questions"]
    run_id = gated.post(
        "/api/runs", json={"brief": BRIEF, "questions": questions}, headers=password
    ).json()["run_id"]
    base = f"/api/runs/{run_id}"
    guarded = [
        ("get", "/api/runs", None),
        ("get", base, None),
        ("get", f"{base}/results", None),
        ("get", f"{base}/answers", None),
        ("get", f"{base}/files/client_report.md", None),
        ("post", f"{base}/ask", {"question": "why did you stop?"}),
        ("post", f"{base}/decision", {"approved": True}),
    ]
    for method, path, body in guarded:
        assert gated.request(method, path, json=body).status_code == 401, path
    wrong = gated.get(base, headers={"X-Demo-Password": "nope"})
    assert wrong.status_code == 401  # one wrong guess: refused, the gate is not locked yet
    # The open endpoints the page needs before it has the password.
    assert gated.get("/api/meta").json()["password_required"] is True
    assert gated.get(base, headers=password).status_code in (200, 409)


def test_asking_without_a_password_does_not_lock_the_gate(gated):
    for _ in range(20):
        assert gated.get("/api/runs").status_code == 401
    ok = gated.post("/api/access", headers={"X-Demo-Password": "s3cret"})
    assert ok.status_code == 200


def test_the_statistics_are_computed_once_per_version_of_the_answers(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from ai_share_of_voice.api import server

    run_dir = tmp_path / "run-20260101-000000-abc123"
    run_dir.mkdir()
    answers = run_dir / "answers.csv"
    answers.write_text("a", encoding="utf-8")
    calls = []

    def fake_run_stats(data, runs_dir):
        calls.append(data.metric)
        row = SimpleNamespace(brand="Glovo", estimate=0.5, ci_low=0.4, ci_high=0.6, ci_method="x")
        return SimpleNamespace(rows=(row,), n_questions=10)

    monkeypatch.setattr(server, "run_stats", fake_run_stats)
    server._compute_metric.cache_clear()

    first = server._metric(tmp_path, run_dir.name, "mention_rate")
    again = server._metric(tmp_path, run_dir.name, "mention_rate")
    assert (
        first
        == again
        == ({"Glovo": {"estimate": 0.5, "ci_low": 0.4, "ci_high": 0.6, "method": "x"}}, 10)
    )
    assert calls == ["mention_rate"]

    answers.write_text("a longer file after a new round", encoding="utf-8")
    server._metric(tmp_path, run_dir.name, "mention_rate")
    assert calls == ["mention_rate", "mention_rate"]  # the file changed: computed again

    assert server._metric(tmp_path, "run-missing", "mention_rate") == ({}, 0)


@pytest.mark.parametrize(
    ("cell", "expected"),
    [("0.25", 0.25), ("", None), (None, None), ("n/a", None), ("nan", None), ("0", 0.0)],
)
def test_a_csv_cell_is_a_number_or_unknown_never_an_invented_zero(cell, expected):
    from ai_share_of_voice.api.server import _number

    assert _number(cell) == expected


def test_too_many_requests_get_429(tmp_path, monkeypatch):
    from ai_share_of_voice.api import server

    monkeypatch.setenv("LLM_MODE", "simulated")
    monkeypatch.setattr(server, "RATE_LIMIT", (3, 60.0))
    settings = load_settings()
    controller = RunController(settings, runs_dir=tmp_path, data_dir=Path("data"))
    limited = TestClient(create_app(settings, controller))
    codes = [limited.get("/api/meta").status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
