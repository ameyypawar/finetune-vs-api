"""finetune_vs_api.lora_merge: the numpy-only LoRA merge the serving fallbacks use, checked against the
formula (W + alpha / r * B @ A) on tiny weights written by hand. Nothing here needs torch or peft."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from finetune_vs_api import lora_merge
from finetune_vs_api.lora_merge import MergeError

RNG = np.random.default_rng(7)


def bf16_exact(values: np.ndarray) -> np.ndarray:
    """float32 values with the low 16 bits zeroed: exactly representable in bfloat16."""
    return (np.asarray(values, dtype=np.float32).view(np.uint32) & 0xFFFF0000).view(np.float32)


def write_safetensors(path, tensors: dict[str, tuple[str, np.ndarray]], metadata=None) -> None:
    """A safetensors file with F32, F16 or BF16 tensors. BF16 values must be exactly representable (see bf16_exact)."""
    header, chunks, offset = {}, [], 0
    for name, (dtype, values) in tensors.items():
        if dtype == "F32":
            raw = np.asarray(values, "<f4").tobytes()
        elif dtype == "F16":
            raw = np.asarray(values, "<f2").tobytes()
        else:
            raw = (np.asarray(values, np.float32).view(np.uint32) >> 16).astype("<u2").tobytes()
        header[name] = {"dtype": dtype, "shape": list(values.shape), "data_offsets": [offset, offset + len(raw)]}
        chunks.append(raw)
        offset += len(raw)
    if metadata:
        header["__metadata__"] = metadata
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks))


def write_adapter(directory, pairs: dict[str, tuple[np.ndarray, np.ndarray]], *, r=2, alpha=4, key_format="base_model.model.{m}.lora_{ab}.weight", **config) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for module, (a, b) in pairs.items():
        tensors[key_format.format(m=module, ab="A")] = ("F32", a)
        tensors[key_format.format(m=module, ab="B")] = ("F32", b)
    write_safetensors(directory / "adapter_model.safetensors", tensors)
    (directory / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": r, "lora_alpha": alpha, **config}))


def read_all(path) -> dict[str, np.ndarray]:
    f = lora_merge.SafetensorsFile(path)
    return {n: f.read(n) for n in f.names()}


@pytest.fixture
def base(tmp_path):
    """A two-shard 'model': q_proj and the embeddings in one shard, down_proj and the norm in the other."""
    directory = tmp_path / "base"
    directory.mkdir()
    q, embed = bf16_exact(RNG.normal(size=(8, 6))), bf16_exact(RNG.normal(size=(10, 6)))
    down, norm = RNG.normal(size=(6, 12)).astype(np.float32), bf16_exact(RNG.normal(size=(6,)))
    write_safetensors(directory / "model-00001-of-00002.safetensors",
                      {"model.layers.0.self_attn.q_proj.weight": ("BF16", q), "model.embed_tokens.weight": ("BF16", embed)}, {"format": "pt"})
    write_safetensors(directory / "model-00002-of-00002.safetensors",
                      {"model.layers.0.mlp.down_proj.weight": ("F32", down), "model.norm.weight": ("BF16", norm)}, {"format": "pt"})
    (directory / "config.json").write_text('{"model_type": "qwen3", "rope_theta": 5000000.0, "torch_dtype": "bfloat16"}')
    (directory / "tokenizer.json").write_text('{"version": "1.0"}')
    (directory / "model.safetensors.index.json").write_text('{"metadata": {"total_size": 1}, "weight_map": {}}')
    (directory / ".cache").mkdir()
    (directory / ".cache" / "ignored.txt").write_text("a directory, not copied")
    return directory, {"q": q, "embed": embed, "down": down, "norm": norm}


@pytest.fixture
def adapter(tmp_path):
    a_q, b_q = RNG.normal(size=(2, 6)).astype(np.float32), RNG.normal(size=(8, 2)).astype(np.float32)
    a_d, b_d = RNG.normal(size=(2, 12)).astype(np.float32), RNG.normal(size=(6, 2)).astype(np.float32)
    write_adapter(tmp_path / "adapter", {"model.layers.0.self_attn.q_proj": (a_q, b_q), "model.layers.0.mlp.down_proj": (a_d, b_d)})
    return tmp_path / "adapter", {"q": (a_q, b_q), "down": (a_d, b_d)}


# --- reading ------------------------------------------------------------------------------------------------------


def test_bfloat16_f16_and_f32_are_read_as_float32(tmp_path):
    values = np.array([[1.0, -2.5, 0.15625], [65280.0, 0.0, -0.0078125]], np.float32)
    for dtype in ("BF16", "F16", "F32"):
        path = tmp_path / f"{dtype}.safetensors"
        write_safetensors(path, {"w": (dtype, values), "empty": (dtype, np.zeros((0, 3), np.float32))})
        f = lora_merge.SafetensorsFile(path)
        assert f.names() == ["w", "empty"]
        np.testing.assert_array_equal(f.read("w"), values)
        assert f.read("w").dtype == np.float32 and f.shape("w") == (2, 3) and f.read("empty").shape == (0, 3)


def test_the_tensors_come_back_in_file_order_with_their_metadata(tmp_path):
    path = tmp_path / "x.safetensors"
    write_safetensors(path, {"b": ("F32", np.ones(2)), "a": ("F32", np.ones(3)), "c": ("F32", np.ones(1))}, {"format": "pt"})
    f = lora_merge.SafetensorsFile(path)
    assert f.names() == ["b", "a", "c"] and f.metadata == {"format": "pt"}


@pytest.mark.parametrize("damage", ["short", "huge_header", "bad_json", "truncated_data", "dtype"])
def test_a_damaged_file_is_refused(tmp_path, damage):
    path = tmp_path / "x.safetensors"
    write_safetensors(path, {"w": ("F32", np.ones((4, 4)))})
    raw = path.read_bytes()
    if damage == "short":
        path.write_bytes(raw[:5])
    elif damage == "huge_header":
        path.write_bytes(struct.pack("<Q", 10**12) + raw[8:])
    elif damage == "bad_json":
        path.write_bytes(struct.pack("<Q", 9) + b"not json!" + raw[8:])
    elif damage == "truncated_data":
        path.write_bytes(raw[:-10])
    else:
        header = {"w": {"dtype": "I64", "shape": [1], "data_offsets": [0, 8]}}
        encoded = json.dumps(header).encode()
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(8))
    with pytest.raises(MergeError):
        lora_merge.SafetensorsFile(path)


# --- the arithmetic -----------------------------------------------------------------------------------------------------


def test_the_merge_is_w_plus_alpha_over_r_times_b_a_in_float16(base, adapter, tmp_path):
    base_dir, weights = base
    adapter_dir, lora = adapter
    report = lora_merge.merge_adapter(base_dir, adapter_dir, tmp_path / "out")
    assert report == {"shards": 2, "tensors": 4, "merged_modules": 2, "scale": 2.0, "rank": 2}  # alpha 4 over r 2
    first = read_all(tmp_path / "out" / "model-00001-of-00002.safetensors")
    second = read_all(tmp_path / "out" / "model-00002-of-00002.safetensors")
    a_q, b_q = lora["q"]
    a_d, b_d = lora["down"]
    expected_q = weights["q"].astype(np.float64) + 2.0 * (b_q.astype(np.float64) @ a_q.astype(np.float64))
    expected_down = weights["down"].astype(np.float64) + 2.0 * (b_d.astype(np.float64) @ a_d.astype(np.float64))
    np.testing.assert_allclose(first["model.layers.0.self_attn.q_proj.weight"], expected_q, rtol=2e-3, atol=2e-3)  # float16 precision
    np.testing.assert_allclose(second["model.layers.0.mlp.down_proj.weight"], expected_down, rtol=2e-3, atol=2e-3)
    assert not np.allclose(first["model.layers.0.self_attn.q_proj.weight"], weights["q"], atol=1e-3)  # it really changed
    # everything else is the base, in float16
    np.testing.assert_array_equal(first["model.embed_tokens.weight"], weights["embed"].astype(np.float16).astype(np.float32))
    np.testing.assert_array_equal(second["model.norm.weight"], weights["norm"].astype(np.float16).astype(np.float32))


def test_the_output_has_the_same_files_names_order_and_only_float16(base, adapter, tmp_path):
    base_dir, _ = base
    lora_merge.merge_adapter(base_dir, adapter[0], tmp_path / "out")
    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == sorted(["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "config.json", "tokenizer.json", "model.safetensors.index.json"])
    for shard in sorted(base_dir.glob("*.safetensors")):
        before, after = lora_merge.SafetensorsFile(shard), lora_merge.SafetensorsFile(out / shard.name)
        assert after.names() == before.names() and after.metadata == {"format": "pt"}
        assert {e["dtype"] for e in after.header.values()} == {"F16"}
        assert [after.shape(n) for n in after.names()] == [before.shape(n) for n in before.names()]
        (header_length,) = struct.unpack("<Q", (out / shard.name).read_bytes()[:8])
        assert (8 + header_length) % 8 == 0  # the data starts on an 8-byte boundary
    for name in ("config.json", "tokenizer.json", "model.safetensors.index.json"):
        assert (out / name).read_bytes() == (base_dir / name).read_bytes()  # copied untouched: the config is the base model's own
    assert not (out / ".cache").exists()


def test_rslora_scales_by_alpha_over_the_square_root_of_r(tmp_path):
    assert lora_merge.lora_scale({"r": 16, "lora_alpha": 32}) == 2.0
    assert lora_merge.lora_scale({"r": 16, "lora_alpha": 32, "use_rslora": True}) == pytest.approx(8.0)
    assert lora_merge.lora_scale({"r": 4, "lora_alpha": 1.5, "use_rslora": False}) == pytest.approx(0.375)


@pytest.mark.parametrize("config", [{}, {"r": 0, "lora_alpha": 1}, {"r": 2.0, "lora_alpha": 1}, {"r": 2}, {"r": 2, "lora_alpha": "4"}, {"r": True, "lora_alpha": 1}])
def test_a_config_without_a_usable_rank_and_alpha_is_refused(config):
    with pytest.raises(MergeError, match="needs an integer r"):
        lora_merge.lora_scale(config)


@pytest.mark.parametrize("key", ["use_dora", "lora_bias", "fan_in_fan_out", "modules_to_save", "rank_pattern", "alpha_pattern"])
def test_adapter_variants_that_are_not_plain_lora_are_refused(key):
    value = {"lm_head": 8} if key.endswith("_pattern") else (["lm_head"] if key == "modules_to_save" else True)
    with pytest.raises(MergeError, match=key):
        lora_merge.lora_scale({"r": 2, "lora_alpha": 4, key: value})
    assert lora_merge.lora_scale({"r": 2, "lora_alpha": 4, key: {} if key.endswith("_pattern") else None}) == 2.0  # empty or unset is plain LoRA


def test_peft_keys_with_an_adapter_name_are_read_too(base, tmp_path):
    base_dir, weights = base
    a, b = RNG.normal(size=(2, 6)).astype(np.float32), RNG.normal(size=(8, 2)).astype(np.float32)
    write_adapter(tmp_path / "named", {"model.layers.0.self_attn.q_proj": (a, b)}, key_format="base_model.model.{m}.lora_{ab}.default.weight")
    report = lora_merge.merge_adapter(base_dir, tmp_path / "named", tmp_path / "out")
    assert report["merged_modules"] == 1
    write_adapter(tmp_path / "bare", {"model.layers.0.self_attn.q_proj": (a, b)}, key_format="{m}.lora_{ab}.weight")  # no prefix at all
    assert lora_merge.merge_adapter(base_dir, tmp_path / "bare", tmp_path / "out2")["merged_modules"] == 1


# --- what is refused -------------------------------------------------------------------------------------------------------------


def test_an_adapter_layer_the_base_does_not_have_is_an_error_not_a_silent_skip(base, tmp_path):
    base_dir, _ = base
    a, b = RNG.normal(size=(2, 6)).astype(np.float32), RNG.normal(size=(8, 2)).astype(np.float32)
    write_adapter(tmp_path / "adapter", {"model.layers.0.self_attn.q_proj": (a, b), "model.layers.7.self_attn.q_proj": (a, b)})
    with pytest.raises(MergeError, match=r"1 adapter modules are not in the base weights, e.g. \['model.layers.7.self_attn.q_proj'\]"):
        lora_merge.merge_adapter(base_dir, tmp_path / "adapter", tmp_path / "out")


def test_shapes_that_do_not_fit_are_refused(base, tmp_path):
    base_dir, _ = base
    write_adapter(tmp_path / "adapter", {"model.layers.0.self_attn.q_proj": (np.ones((2, 5), np.float32), np.ones((8, 2), np.float32))})
    with pytest.raises(MergeError, match="do not fit a weight of shape"):
        lora_merge.merge_adapter(base_dir, tmp_path / "adapter", tmp_path / "out")
    write_adapter(tmp_path / "rank", {"model.layers.0.self_attn.q_proj": (np.ones((3, 6), np.float32), np.ones((8, 2), np.float32))})
    with pytest.raises(MergeError, match="do not fit a weight of shape"):
        lora_merge.merge_adapter(base_dir, tmp_path / "rank", tmp_path / "out")


def test_an_adapter_with_stray_or_half_tensors_is_refused(base, tmp_path):
    base_dir, _ = base
    (tmp_path / "stray").mkdir()
    write_safetensors(tmp_path / "stray" / "adapter_model.safetensors", {"base_model.model.lm_head.weight": ("F32", np.ones((2, 2)))})
    (tmp_path / "stray" / "adapter_config.json").write_text('{"r": 2, "lora_alpha": 4}')
    with pytest.raises(MergeError, match="is not a LoRA A or B weight"):
        lora_merge.merge_adapter(base_dir, tmp_path / "stray", tmp_path / "out")
    (tmp_path / "half").mkdir()
    write_safetensors(tmp_path / "half" / "adapter_model.safetensors", {"base_model.model.model.norm.lora_A.weight": ("F32", np.ones((2, 6)))})
    (tmp_path / "half" / "adapter_config.json").write_text('{"r": 2, "lora_alpha": 4}')
    with pytest.raises(MergeError, match="has only lora_A"):
        lora_merge.merge_adapter(base_dir, tmp_path / "half", tmp_path / "out")
    (tmp_path / "empty").mkdir()
    write_safetensors(tmp_path / "empty" / "adapter_model.safetensors", {})
    (tmp_path / "empty" / "adapter_config.json").write_text('{"r": 2, "lora_alpha": 4}')
    with pytest.raises(MergeError, match="holds no LoRA weights"):
        lora_merge.merge_adapter(base_dir, tmp_path / "empty", tmp_path / "out")


def test_missing_files_are_reported(base, adapter, tmp_path):
    with pytest.raises(MergeError, match="cannot read"):
        lora_merge.merge_adapter(base[0], tmp_path / "no-adapter", tmp_path / "out")
    (tmp_path / "bare").mkdir()
    with pytest.raises(MergeError, match="no .safetensors files"):
        lora_merge.merge_adapter(tmp_path / "bare", adapter[0], tmp_path / "out")


def test_a_value_that_does_not_fit_float16_is_an_overflow_error_not_an_infinity(tmp_path):
    big = tmp_path / "big"
    big.mkdir()
    weight = np.full((4, 4), 60000.0, np.float32)  # fits float16 (max 65504) on its own
    write_safetensors(big / "m.safetensors", {"model.layers.0.self_attn.q_proj.weight": ("F32", weight)})
    a, b = np.ones((1, 4), np.float32), np.full((4, 1), 4000.0, np.float32)
    write_adapter(tmp_path / "adapter", {"model.layers.0.self_attn.q_proj": (a, b)}, r=1, alpha=2)  # + 2 * 4000 = 68000: over
    with pytest.raises(MergeError, match=r"fp16 overflow"):
        lora_merge.merge_adapter(big, tmp_path / "adapter", tmp_path / "out")
    small = np.full((4, 1), 400.0, np.float32)  # 60000 + 800 = 60800: fits
    write_adapter(tmp_path / "ok", {"model.layers.0.self_attn.q_proj": (a, small)}, r=1, alpha=2)
    lora_merge.merge_adapter(big, tmp_path / "ok", tmp_path / "out2")
    np.testing.assert_allclose(read_all(tmp_path / "out2" / "m.safetensors")["model.layers.0.self_attn.q_proj.weight"], 60800.0, rtol=1e-3)


def test_a_base_weight_that_is_not_finite_is_refused(tmp_path):
    bad = tmp_path / "bad"
    bad.mkdir()
    write_safetensors(bad / "m.safetensors", {"model.norm.weight": ("F32", np.array([1.0, np.nan], np.float32)), "model.layers.0.self_attn.q_proj.weight": ("F32", np.ones((2, 2)))})
    a, b = np.ones((1, 2), np.float32), np.ones((2, 1), np.float32)
    write_adapter(tmp_path / "adapter", {"model.layers.0.self_attn.q_proj": (a, b)}, r=1, alpha=1)
    with pytest.raises(MergeError, match="model.norm.weight: not finite"):
        lora_merge.merge_adapter(bad, tmp_path / "adapter", tmp_path / "out")


# --- against PEFT itself (runs only where torch, transformers and peft are installed) ----------------------------------------------


def test_the_merge_equals_pefts_merge_and_unload_on_a_tiny_qwen3(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    if not hasattr(transformers, "Qwen3Config"):
        pytest.skip("this transformers has no Qwen3")

    def load(path, dtype):
        try:
            return transformers.Qwen3ForCausalLM.from_pretrained(path, dtype=dtype)
        except TypeError:  # before the dtype= spelling
            return transformers.Qwen3ForCausalLM.from_pretrained(path, torch_dtype=dtype)

    torch.manual_seed(0)
    config = transformers.Qwen3Config(
        hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        vocab_size=320, max_position_embeddings=128, tie_word_embeddings=True,
    )
    base_dir, adapter_dir = tmp_path / "base", tmp_path / "adapter"
    transformers.Qwen3ForCausalLM(config).to(torch.bfloat16).save_pretrained(base_dir, max_shard_size="150KB")  # several shards, bfloat16 like the real one
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    tuned = peft.get_peft_model(load(base_dir, torch.float32), peft.LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, bias="none", target_modules=targets))
    with torch.no_grad():
        for name, parameter in tuned.named_parameters():
            if "lora_" in name:
                parameter.copy_(torch.randn_like(parameter) * 0.1)  # B starts at zero; a zero delta would prove nothing
    tuned.save_pretrained(adapter_dir)  # PEFT's own key names and config

    expected = peft.PeftModel.from_pretrained(load(base_dir, torch.float16), adapter_dir).merge_and_unload().state_dict()
    report = lora_merge.merge_adapter(base_dir, adapter_dir, tmp_path / "mine")
    mine = load(tmp_path / "mine", torch.float16).state_dict()  # and the result loads like any model
    untouched = load(base_dir, torch.float16).state_dict()

    assert report["merged_modules"] == 3 * len(targets) and report["scale"] == 2.0
    assert set(expected) == set(mine)
    changed = 0
    for name in expected:
        assert (expected[name].float() - mine[name].float()).abs().max().item() < 5e-3, name
        changed += int((expected[name].float() - untouched[name].float()).abs().max().item() > 1e-3)
    assert changed == 3 * len(targets)  # the adapter really changed the weights it targets, and nothing else
