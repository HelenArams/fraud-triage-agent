from __future__ import annotations

import json

import pytest

from fraud_triage.features import compute
from fraud_triage.llm.base import tool_result
from fraud_triage.llm.offline import OfflineAnalyst
from fraud_triage.scoring import score
from fraud_triage.tools import TOOL_SPECS, ToolContext, ToolError, case_library, run_tool

from factories import T0, make_txn, minutes


def ctx_for(target_kw: dict[str, object], hist_kw: list[dict[str, object]]) -> ToolContext:
    hist = [
        make_txn(id=f"h{i}", ts=T0 - minutes(60 * (i + 1)), **kw) for i, kw in enumerate(hist_kw)
    ]
    txn = make_txn(id="t_target", **target_kw)
    feats = compute(txn, hist)
    return ToolContext(txn=txn, history=hist, features=feats, rule_score=score(feats))


def ato_ctx() -> ToolContext:
    known = {"channel": "ecom", "three_ds_result": "Y", "device_id": "dev-1", "ip_country": "DE"}
    return ctx_for(
        {
            "channel": "ecom",
            "three_ds_result": "N",
            "device_id": "dev-x",
            "ip_country": "RO",
            "amount_minor": 99000,
            "merchant_id": "m_curry_online",
        },
        [{**known, "amount_minor": a} for a in (2000, 2500, 3000, 2200)],
    )


def test_case_library_loads() -> None:
    ids = [c.id for c in case_library()]
    assert len(ids) == len(set(ids)) >= 8


def test_tool_specs_have_unique_names_and_schemas() -> None:
    names = [t.name for t in TOOL_SPECS]
    assert len(names) == len(set(names))
    assert all(t.input_schema["type"] == "object" for t in TOOL_SPECS)


def test_card_history_newest_first_and_no_descriptors() -> None:
    out = json.loads(run_tool(ato_ctx(), "card_history", {"limit": 2}))
    assert [h["id"] for h in out] == ["h0", "h1"]
    assert all("merchant_name" not in h for h in out)


def test_merchant_profile_known_and_unknown() -> None:
    ctx = ato_ctx()
    known = json.loads(run_tool(ctx, "merchant_profile", {"merchant_id": "m_curry_online"}))
    assert known["known"] and known["country"] == "GB"
    unknown = json.loads(run_tool(ctx, "merchant_profile", {"merchant_id": "m_nope"}))
    assert unknown == {"merchant_id": "m_nope", "known": False, "seen_on_this_card": 0}


def test_similar_cases_ranks_takeover_cases_first() -> None:
    out = json.loads(run_tool(ato_ctx(), "similar_cases", {"k": 2}))
    assert out[0]["case_id"] == "C-103"
    assert out[0]["outcome"] == "fraud"


def test_similar_cases_empty_when_nothing_fired() -> None:
    assert json.loads(run_tool(ctx_for({}, []), "similar_cases", {})) == []


def test_bad_tool_calls() -> None:
    with pytest.raises(ToolError):
        run_tool(ato_ctx(), "drop_tables", {})
    with pytest.raises(ToolError):
        run_tool(ato_ctx(), "card_history", {"card_token": "tok_someone_else"})


def test_known_refs() -> None:
    refs = ato_ctx().known_refs()
    assert {"feature:new_device", "case:C-101", "txn:h0", "txn:t_target", "rule_score"} <= refs
    assert "case:C-999" not in refs


def test_offline_analyst_round_trip() -> None:
    ctx = ato_ctx()
    analyst = OfflineAnalyst()
    msgs: list[dict[str, object]] = [{"role": "user", "content": "triage t_target"}]
    first = analyst.create(system="", tools=TOOL_SPECS, messages=msgs)
    assert [c.name for c in first.tool_calls] == ["feature_breakdown", "similar_cases"]
    msgs += [
        {"role": "assistant", "content": first.content},
        {
            "role": "user",
            "content": [
                tool_result(c.id, run_tool(ctx, c.name, c.input)) for c in first.tool_calls
            ],
        },
    ]
    [submit] = analyst.create(system="", tools=TOOL_SPECS, messages=msgs).tool_calls
    assert submit.name == "submit_decision"
    assert submit.input["verdict"] == "BLOCK_CARD"
    assert "case:C-103" in submit.input["evidence"]
