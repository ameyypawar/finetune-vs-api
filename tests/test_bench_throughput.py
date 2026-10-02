"""scripts/bench_throughput.py: the arithmetic, the operating point, the cost, and the whole run
against a stub OpenAI-compatible server on a local socket. The `openai` package is not a
dependency of this repository, so most tests send through httpx; the script's own path through
`openai` is exercised with a stand-in module that has the real client's interface."""

from __future__ import annotations

import asyncio
import json
import random
import types

import pytest
import yaml

from conftest import ROOT, load_script
from finetune_vs_api import cost, metrics
from serving_helpers import HttpxSender, StubChatServer, install_fake_openai

bench = load_script("bench_throughput")
RENTALS = yaml.safe_load((ROOT / "configs" / "sources.yaml").read_text())["gpu_rental"]
ON_DEMAND, SPOT = 0.526, 0.274


def raw(latencies, *, concurrency=8, wall=2.0, tokens=None, errors=0, requests=None, finish=None):
    level = bench.LevelRaw(concurrency, requests if requests is not None else len(latencies) + errors)
    level.latencies, level.wall_s, level.errors = list(latencies), wall, errors
    level.completion_tokens = list(tokens if tokens is not None else [10] * len(latencies))
    level.prompt_tokens = [40] * len(latencies)
    level.finish_reasons = dict(finish or {"stop": len(latencies)})
    return level


def level(concurrency, p95, rate=5.0, errors=0):
    """A summarized level with a chosen p95 and request rate."""
    latencies = [p95] * 20
    return bench.summarize_level(raw(latencies, concurrency=concurrency, wall=len(latencies) / rate, errors=errors), 1.0)


# --- the numbers -------------------------------------------------------------------------------------------


def test_percentile_is_the_nearest_rank_rule_the_run_summaries_use():
    rng = random.Random(7)
    for n in (1, 2, 5, 10, 19, 20, 100, 1000):
        values = [rng.random() for _ in range(n)]
        for q in (0, 1, 50, 90, 95, 99, 100):
            assert bench.percentile(values, q) == metrics.percentile(values, q)
    assert bench.percentile(range(1, 11), 95) == 10 and bench.percentile(range(1, 21), 95) == 19
    with pytest.raises(ValueError):
        bench.percentile([], 50)


def test_a_levels_numbers():
    latencies = [0.1 * i for i in range(1, 11)]  # 0.1 .. 1.0
    summary = bench.summarize_level(raw(latencies, wall=2.0, tokens=[10] * 10), 1.0)
    assert (summary["ok"], summary["errors"], summary["requests"], summary["concurrency"]) == (10, 0, 10, 8)
    assert summary["requests_per_s"] == pytest.approx(5.0) and summary["output_tokens_per_s"] == pytest.approx(50.0)
    lat = summary["latency_s"]
    assert (lat["p50"], lat["p95"], lat["max"], lat["n"], lat["method"]) == (0.5, 1.0, 1.0, 10, "nearest-rank")
    assert lat["mean"] == pytest.approx(0.55)
    assert summary["mean_output_tokens"] == 10 and summary["mean_prompt_tokens"] == 40
    assert summary["finish_reasons"] == {"stop": 10} and summary["meets_p95_limit"] is True


def test_failed_requests_are_counted_but_do_not_make_the_rate_or_the_latency():
    summary = bench.summarize_level(raw([0.2] * 8, wall=4.0, errors=2), 1.0)
    assert (summary["ok"], summary["errors"], summary["requests"]) == (8, 2, 10)
    assert summary["error_rate"] == pytest.approx(0.2)
    assert summary["requests_per_s"] == pytest.approx(2.0)  # successes only
    assert summary["meets_p95_limit"] is False  # a level with a failure cannot be the operating point


def test_a_level_where_everything_failed_has_no_latency_and_no_rate():
    summary = bench.summarize_level(raw([], wall=1.0, errors=5), 1.0)
    assert summary["latency_s"] is None and summary["requests_per_s"] == 0.0 and summary["meets_p95_limit"] is False
    assert summary["output_tokens_per_s"] is None and summary["mean_output_tokens"] is None


def test_tokens_per_second_is_none_when_the_server_reported_no_usage():
    summary = bench.summarize_level(raw([0.1] * 4, tokens=[]), 1.0)
    assert summary["output_tokens_per_s"] is None and summary["requests_per_s"] > 0


# --- the operating point ---------------------------------------------------------------------------------------


def test_the_operating_point_is_the_highest_concurrency_with_p95_at_most_one_second():
    levels = [level(1, 0.30), level(8, 0.90), level(32, 1.40), level(64, 3.00)]
    point = bench.operating_point(levels, 1.0)
    assert point["concurrency"] == 8
    assert point["latency_s"] == {"p50": 0.90, "p95": 0.90} and point["requests_per_s"] == pytest.approx(5.0)
    assert "1 s" in point["rule"] and "highest concurrency" in point["rule"]


def test_a_p95_of_exactly_the_limit_qualifies_and_the_rule_is_the_highest_level_that_qualifies():
    assert bench.operating_point([level(1, 1.0), level(8, 1.0001)], 1.0)["concurrency"] == 1
    # not "stop at the first level over the limit": the highest level that meets it is the point
    assert bench.operating_point([level(1, 0.2), level(8, 1.5), level(32, 0.9)], 1.0)["concurrency"] == 32


def test_a_level_with_a_failed_request_is_never_the_operating_point():
    levels = [level(1, 0.2), level(8, 0.4, errors=1), level(32, 5.0)]
    assert bench.operating_point(levels, 1.0)["concurrency"] == 1


def test_no_level_qualifying_gives_no_operating_point_rather_than_a_guess():
    assert bench.operating_point([level(1, 1.2), level(8, 2.0)], 1.0) is None
    assert bench.operating_point([], 1.0) is None


# --- the price key and the cost ----------------------------------------------------------------------------------


def test_the_price_key_names_the_rental_and_the_billing_option():
    assert bench.resolve_price_key(RENTALS, "aws_g4dn_xlarge_ondemand") == ("aws-g4dn.xlarge", "on_demand")
    assert bench.resolve_price_key(RENTALS, "aws_g4dn_xlarge_spot") == ("aws-g4dn.xlarge", "spot")
    assert bench.resolve_price_key(RENTALS, "AWS-g4dn.xlarge-on-demand") == ("aws-g4dn.xlarge", "on_demand")
    assert bench.known_price_keys(RENTALS) == ["aws_g4dn_xlarge_ondemand", "aws_g4dn_xlarge_spot"]
    for bad in ("aws_g4dn_xlarge", "aws_g4dn_xlarge_reserved", "gcp_t4_ondemand", ""):
        with pytest.raises(bench.BenchError, match="known keys: aws_g4dn_xlarge_ondemand"):
            bench.resolve_price_key(RENTALS, bad)


def test_the_sources_file_has_the_prices_the_cost_uses():
    entry = RENTALS["aws-g4dn.xlarge"]["usd_per_hour"]
    assert (entry["on_demand"], entry["spot"]) == (ON_DEMAND, SPOT)


def test_cost_per_1k_calls_is_the_repositorys_selfhost_formula_for_both_prices():
    point = bench.operating_point([level(1, 0.3, rate=2.0), level(8, 0.9, rate=5.0)], 1.0)
    result = bench.selfhost_costs(RENTALS, "aws_g4dn_xlarge_ondemand", [level(1, 0.3, rate=2.0), level(8, 0.9, rate=5.0)], point)
    at = result["at_operating_point"]
    assert at["concurrency"] == 8 and at["requests_per_s"] == pytest.approx(5.0)
    assert at["per_1k_calls_usd"]["on_demand"] == pytest.approx(ON_DEMAND / (3600 * 5.0) * 1000)
    assert at["per_1k_calls_usd"]["spot"] == pytest.approx(SPOT / (3600 * 5.0) * 1000)
    assert at["per_1k_calls_usd"]["on_demand"] == cost.selfhost_per_1k(ON_DEMAND, at["requests_per_s"])
    assert at["headline_billing"] == "on_demand" and at["headline_usd"] == at["per_1k_calls_usd"]["on_demand"]
    assert result["by_concurrency"]["1"]["on_demand"] == pytest.approx(ON_DEMAND / (3600 * 2.0) * 1000)
    assert result["rental"]["id"] == "aws-g4dn.xlarge" and result["rental"]["usd_per_hour"] == {"on_demand": ON_DEMAND, "spot": SPOT}
    assert result["rental"]["url"].startswith("https://") and result["rental"]["retrieved_on"] == "2026-10-01"
    assert "fully busy" in result["assumes"] and "selfhost_per_1k" in result["formula"]


def test_the_spot_key_headlines_the_spot_price_and_a_missing_operating_point_leaves_no_headline_cost():
    point = bench.operating_point([level(8, 0.5, rate=4.0)], 1.0)
    spot = bench.selfhost_costs(RENTALS, "aws_g4dn_xlarge_spot", [level(8, 0.5, rate=4.0)], point)
    assert spot["headline_billing"] == "spot" and spot["at_operating_point"]["headline_usd"] == spot["at_operating_point"]["per_1k_calls_usd"]["spot"]
    none = bench.selfhost_costs(RENTALS, "aws_g4dn_xlarge_ondemand", [level(8, 2.5, rate=4.0)], None)
    assert none["at_operating_point"] is None and "8" in none["by_concurrency"]  # per-level costs are still informative
    dead = bench.selfhost_costs(RENTALS, "aws_g4dn_xlarge_ondemand", [bench.summarize_level(raw([], errors=3, wall=1.0), 1.0)], None)
    assert dead["by_concurrency"] == {}  # a rate of zero has no cost per call


# --- reading the prompts ------------------------------------------------------------------------------------------


def test_prompt_files(tmp_path):
    path = tmp_path / "p.jsonl"
    messages = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "wake me at six"}]
    path.write_text(json.dumps({"messages": messages}) + "\n\n" + json.dumps({"text": "play jazz"}) + "\n")
    assert bench.load_prompts(path) == [messages, [{"role": "user", "content": "play jazz"}]]
    txt = tmp_path / "p.txt"
    txt.write_text("one\n\ntwo \n")
    assert bench.load_prompts(txt) == [[{"role": "user", "content": "one"}], [{"role": "user", "content": "two"}]]


@pytest.mark.parametrize(
    "content",
    ["", "\n\n", "not json\n", '{"foo": 1}\n', '{"messages": []}\n', '{"messages": [{"role": "user"}]}\n', '{"text": "  "}\n', "[1, 2]\n"],
)
def test_bad_prompt_files_are_refused(tmp_path, content):
    path = tmp_path / "p.jsonl"
    path.write_text(content)
    with pytest.raises(bench.BenchError):
        bench.load_prompts(path)
    with pytest.raises(bench.BenchError, match="cannot read"):
        bench.load_prompts(tmp_path / "missing.jsonl")


def test_levels_and_labels():
    assert bench.parse_levels("1,8,32,64") == (1, 8, 32, 64) and bench.parse_levels("8, 1,8") == (1, 8)
    for bad in ("", "0", "1,x", "-3"):
        with pytest.raises(bench.BenchError):
            bench.parse_levels(bad)
    assert bench.safe_label("T4") == "T4" and bench.safe_label(" Tesla T4/x ") == "Tesla-T4-x"
    with pytest.raises(bench.BenchError):
        bench.safe_label("///")


# --- one level against a fake sender: exactly the asked concurrency -----------------------------------------------------


class SlowSender:
    """A sender that takes a moment, and remembers how many calls overlapped."""

    def __init__(self, delay=0.01, fail_on=()):
        self.delay, self.fail_on, self.calls, self.inflight, self.max_inflight = delay, set(fail_on), [], 0, 0

    async def __call__(self, messages, max_tokens):
        self.calls.append((messages[-1]["content"], max_tokens))
        number = len(self.calls)
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.inflight -= 1
        if number in self.fail_on:
            raise RuntimeError(f"boom {number}")
        return types.SimpleNamespace(completion_tokens=7, prompt_tokens=30, finish_reason="stop")


def test_a_level_keeps_exactly_the_asked_number_of_requests_in_flight():
    sender = SlowSender()
    prompts = [[{"role": "user", "content": f"q{i}"}] for i in range(3)]
    result = asyncio.run(bench.run_level(sender, prompts, 4, 20, max_tokens=64))
    assert sender.max_inflight == 4 and len(sender.calls) == 20
    assert (len(result.latencies), result.errors) == (20, 0) and all(lat >= 0.01 for lat in result.latencies)
    assert {m for _, m in sender.calls} == {64}
    assert [c for c, _ in sender.calls[:6]] == ["q0", "q1", "q2", "q0", "q1", "q2"]  # prompts are cycled in order
    assert result.completion_tokens == [7] * 20 and result.finish_reasons == {"stop": 20}


def test_failures_are_recorded_and_do_not_stop_the_level():
    sender = SlowSender(delay=0.0, fail_on={2, 5})
    result = asyncio.run(bench.run_level(sender, [[{"role": "user", "content": "q"}]], 2, 10, max_tokens=8))
    assert result.errors == 2 and len(result.latencies) == 8 and len(sender.calls) == 10
    assert result.error_samples == ["RuntimeError: boom 2", "RuntimeError: boom 5"]


def test_fewer_requests_than_the_concurrency_is_fine():
    sender = SlowSender()
    result = asyncio.run(bench.run_level(sender, [[{"role": "user", "content": "q"}]], 8, 3, max_tokens=8))
    assert len(result.latencies) == 3 and sender.max_inflight == 3


def test_a_dead_server_is_found_at_warm_up_and_nothing_is_swept():
    sender = SlowSender(delay=0.0, fail_on=set(range(1, 100)))
    with pytest.raises(bench.BenchError, match="none of 8 warm-up requests"):
        asyncio.run(bench.run_sweep(sender, [[{"role": "user", "content": "q"}]], (1, 8), 10, max_tokens=8, p95_limit_s=1.0, out=lambda m: None))
    assert len(sender.calls) == 8


# --- the whole run, against a stub server on a real socket --------------------------------------------------------------


@pytest.fixture(scope="module")
def prompts_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("prompts") / "throughput_prompts.jsonl"
    rows = [{"messages": [{"role": "system", "content": "Convert the request."}, {"role": "user", "content": f"wake me at {i} please"}]} for i in range(12)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def run_main(argv, server, **overrides):
    lines: list[str] = []
    kw = {"sender_factory": HttpxSender, "out": lines.append, "gpu_probe": lambda: {"name": "Tesla T4", "memory_mib": 15360, "driver": "550.1"}}
    code = bench.main(["--base-url", server.url, *argv], **{**kw, **overrides})
    return code, "\n".join(lines)


def test_a_full_run_writes_the_file_compare_py_reads(tmp_path, prompts_file):
    out = tmp_path / "results" / "serving" / "T4.json"
    with StubChatServer(delay_s=0.01, capacity=4, completion_tokens=25, version="0.11.2") as server:
        code, text = run_main(
            ["--model", "ft-qwen3-4b-lora", "--prompts", str(prompts_file), "--concurrency", "1,4,16", "--requests", "24",
             "--out", str(out), "--dtype", "float16", "--extra", "serving_path=vllm-lora", "--extra", "note=a=b"], server)
        assert code == 0 and "operating point:" in text and str(out) in text
        assert len(server.bodies) == 8 + 3 * 24  # warm-up plus every level
        sent = server.bodies[0]
        assert sent["model"] == "ft-qwen3-4b-lora" and sent["temperature"] == 0 and sent["max_tokens"] == 256
        assert server.max_inflight <= 4  # the stub's own capacity: more requests than that waited their turn
    doc = json.loads(out.read_text())
    # the shape scripts/compare.py documents
    assert doc["gpu"] == "Tesla T4" and doc["gpu_label"] == "T4" and doc["system"] == "ft-qwen3-4b-lora"
    assert [lv["concurrency"] for lv in doc["levels"]] == [1, 4, 16]
    for lv in doc["levels"]:
        assert lv["ok"] == 24 and lv["errors"] == 0 and lv["requests_per_s"] > 0
        assert 0 < lv["latency_s"]["p50"] <= lv["latency_s"]["p95"] and lv["latency_s"]["p95"] >= 0.01
        assert lv["output_tokens_per_s"] == pytest.approx(25 * lv["requests_per_s"]) and lv["mean_output_tokens"] == 25
    assert doc["operating_point"]["concurrency"] == 16  # a stub that serves 4 at a time keeps p95 far below a second
    op = next(lv for lv in doc["levels"] if lv["concurrency"] == doc["operating_point"]["concurrency"])
    assert op["latency_s"]["p95"] <= 1.0 and doc["operating_point"]["requests_per_s"] == op["requests_per_s"]
    # what was measured, and how
    assert (doc["engine"], doc["engine_version"], doc["vllm_version"], doc["dtype"]) == ("vllm", "0.11.2", "0.11.2", "float16")
    assert doc["gpu_memory_mib"] == 15360 and doc["gpu_driver"] == "550.1" and doc["model"] == "ft-qwen3-4b-lora"
    assert doc["workload"]["n_prompts"] == 12 and doc["workload"]["requests_per_level"] == 24 and len(doc["workload"]["prompts_sha256"]) == 64
    assert doc["workload"]["max_tokens"] == 256 and doc["workload"]["temperature"] == 0
    assert doc["extra"] == {"serving_path": "vllm-lora", "note": "a=b"} and doc["p95_limit_s"] == 1.0
    assert doc["created_at"].endswith("Z") and doc["schema_version"] == 1
    at = doc["cost"]["at_operating_point"]
    assert at["per_1k_calls_usd"]["on_demand"] == cost.selfhost_per_1k(ON_DEMAND, doc["operating_point"]["requests_per_s"])
    assert at["per_1k_calls_usd"]["spot"] == cost.selfhost_per_1k(SPOT, doc["operating_point"]["requests_per_s"])
    assert doc["cost"]["price_key"] == "aws_g4dn_xlarge_ondemand" and doc["cost"]["headline_billing"] == "on_demand"
    json.dumps(doc)  # plain JSON all the way down


def test_the_report_passes_comparepys_own_reader(tmp_path, prompts_file):
    """The reader is the results pipeline's; if it cannot be loaded here the check is skipped, never faked."""
    try:
        compare = load_script("compare")
        normalize = compare.normalize_benchmark
    except Exception as exc:  # scripts/compare.py belongs to another part of the repository
        pytest.skip(f"scripts/compare.py cannot be loaded here: {exc}")
    out = tmp_path / "T4.json"
    with StubChatServer(version="0.11.2") as server:
        code, _ = run_main(["--model", "ft-qwen3-4b-lora", "--prompts", str(prompts_file), "--concurrency", "1,8", "--requests", "10", "--out", str(out)], server)
    assert code == 0
    entry = normalize(json.loads(out.read_text()), "default")["ft-qwen3-4b-lora"]
    assert entry["problems"] == [] and entry["single_stream"]["concurrency"] == 1
    assert entry["operating_point"]["concurrency"] in (1, 8) and entry["operating_point"]["requests_per_s"] > 0


def test_a_slow_server_has_no_operating_point_and_says_so(tmp_path, prompts_file):
    out = tmp_path / "T4.json"
    with StubChatServer(delay_s=0.06) as server:
        code, text = run_main(["--model", "m", "--prompts", str(prompts_file), "--concurrency", "1,2", "--requests", "4",
                               "--warmup", "1", "--p95-limit", "0.01", "--out", str(out)], server)
    doc = json.loads(out.read_text())
    assert code == 0 and "operating point: none" in text
    assert doc["operating_point"] is None and "no concurrency level" in doc["operating_point_note"]
    assert doc["cost"]["at_operating_point"] is None and set(doc["cost"]["by_concurrency"]) == {"1", "2"}


def test_failed_requests_are_reported_and_disqualify_the_level(tmp_path, prompts_file):
    out = tmp_path / "T4.json"
    with StubChatServer(fail_every=4) as server:
        code, text = run_main(["--model", "m", "--prompts", str(prompts_file), "--concurrency", "2", "--requests", "20",
                               "--warmup", "0", "--out", str(out)], server)
    doc = json.loads(out.read_text())
    only = doc["levels"][0]
    assert code == 0 and only["errors"] == 5 and only["ok"] == 15 and only["error_rate"] == pytest.approx(0.25)
    assert only["error_samples"] and "500" in only["error_samples"][0] and doc["operating_point"] is None


def test_engine_and_gpu_are_taken_from_flags_when_given_and_detected_otherwise(tmp_path, prompts_file):
    out = tmp_path / "T4.json"
    with StubChatServer() as server:  # answers no /version
        code, _ = run_main(["--model", "m", "--prompts", str(prompts_file), "--concurrency", "1", "--requests", "2", "--warmup", "0",
                            "--out", str(out), "--engine", "llama.cpp", "--engine-version", "b7200", "--gpu-name", "NVIDIA Tesla T4", "--system", "ft-qwen3-4b-lora"], server)
    doc = json.loads(out.read_text())
    assert (doc["engine"], doc["engine_version"], doc["vllm_version"], doc["gpu"], doc["gpu_label"]) == ("llama.cpp", "b7200", None, "NVIDIA Tesla T4", "T4")
    assert doc["dtype"] is None  # the server does not say, so unless it is passed it is not guessed
    with StubChatServer(version="0.11.2") as server:
        assert bench.detect_engine(server.url) == {"engine": "vllm", "version": "0.11.2"}
    with StubChatServer() as server:
        assert bench.detect_engine(server.url) == {"engine": None, "version": None}
    assert bench.detect_engine("http://127.0.0.1:9/v1") == {"engine": None, "version": None}  # nothing listening


def test_bad_options_exit_2_and_write_nothing(tmp_path, prompts_file):
    out = tmp_path / "T4.json"
    with StubChatServer() as server:
        for argv in (
            ["--prompts", str(prompts_file), "--concurrency", "0"],
            ["--prompts", str(prompts_file), "--requests", "0"],
            ["--prompts", str(prompts_file), "--price-key", "gcp_t4"],
            ["--prompts", str(tmp_path / "missing.jsonl")],
            ["--prompts", str(prompts_file), "--extra", "novalue"],
            ["--prompts", str(prompts_file), "--gpu", "///"],
        ):
            code, text = run_main(["--model", "m", "--out", str(out), "--warmup", "0", *argv], server)
            assert code == 2 and "error:" in text, argv
        assert server.bodies == []  # nothing was sent for a bad option
    assert not out.exists()


def test_a_server_that_answers_nothing_exits_2_without_writing(tmp_path, prompts_file):
    out = tmp_path / "T4.json"
    lines: list[str] = []
    code = bench.main(["--base-url", "http://127.0.0.1:9/v1", "--model", "m", "--prompts", str(prompts_file), "--out", str(out), "--requests", "4"],
                      sender_factory=HttpxSender, out=lines.append)
    assert code == 2 and "none of 8 warm-up requests" in "\n".join(lines) and not out.exists()


def test_the_default_output_is_results_serving_gpu_json(monkeypatch, tmp_path):
    assert bench.default_out_path("T4") == ROOT / "results" / "serving" / "T4.json"
    assert bench.default_out_path("Tesla T4") == ROOT / "results" / "serving" / "Tesla-T4.json"
    assert bench.default_sources_path() == ROOT / "configs" / "sources.yaml"


# --- the script's own client: the openai package ---------------------------------------------------------------------------


def test_the_default_sender_uses_the_openai_package_without_retries_and_at_temperature_zero(monkeypatch, tmp_path, prompts_file):
    constructed = install_fake_openai(monkeypatch)
    out = tmp_path / "T4.json"
    lines: list[str] = []
    with StubChatServer(completion_tokens=11, version="0.11.2") as server:
        code = bench.main(["--base-url", server.url, "--model", "ft-qwen3-4b-lora", "--prompts", str(prompts_file), "--concurrency", "2", "--requests", "6",
                           "--warmup", "2", "--out", str(out), "--max-tokens", "64", "--timeout", "9"], out=lines.append, gpu_probe=dict)
        assert code == 0
        assert all(b["temperature"] == 0 and b["max_tokens"] == 64 and b["model"] == "ft-qwen3-4b-lora" for b in server.bodies)
    assert constructed == [{"base_url": server.url, "api_key": "EMPTY", "timeout": 9.0, "max_retries": 0}]
    doc = json.loads(out.read_text())
    assert doc["levels"][0]["mean_output_tokens"] == 11 and doc["gpu"] == "T4"  # no nvidia-smi: the label stands in


def test_without_the_openai_package_the_script_says_so(monkeypatch, tmp_path, prompts_file):
    monkeypatch.setitem(__import__("sys").modules, "openai", None)  # makes `import openai` raise ImportError
    lines: list[str] = []
    code = bench.main(["--base-url", "http://127.0.0.1:9/v1", "--model", "m", "--prompts", str(prompts_file), "--out", str(tmp_path / "x.json")], out=lines.append)
    assert code == 2 and "openai package is required" in "\n".join(lines)


def test_the_script_needs_only_the_openai_package_and_the_standard_library():
    import ast

    tree = ast.parse((ROOT / "scripts" / "bench_throughput.py").read_text())
    top = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            top |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            top.add(node.module.split(".")[0])
    stdlib = set(__import__("sys").stdlib_module_names)
    assert top - stdlib == {"finetune_vs_api", "openai", "yaml"}  # this repo's cost module, the client, and the price file's parser
