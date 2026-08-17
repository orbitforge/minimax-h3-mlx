"""MLX-free contracts for the canonical MiniMax-H3 conditioning artifact."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from minimax_h3_mlx.conditioning_artifact import (
    ARTIFACT_SCHEMA_VERSION,
    CONDITIONING_WIDTH,
    ConditioningArtifactError,
    _artifact_identity,
    bfloat16_bits_from_float32,
    bfloat16_bits_to_float32,
    build_encoder_provenance,
    create_conditioning_artifact,
    create_conditioning_artifact_from_bits,
    load_conditioning_artifact,
    validate_conditioning_artifact,
)


def _checkpoint(root: Path) -> dict:
    (root / "text_encoder").mkdir(parents=True)
    (root / "tokenizer").mkdir()
    (root / "processor").mkdir()
    (root / "text_encoder" / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_vl",
                "text_config": {"num_hidden_layers": 64, "hidden_size": CONDITIONING_WIDTH},
            }
        )
    )
    (root / "tokenizer" / "tokenizer_config.json").write_text('{"add_bos_token":false}')
    (root / "processor" / "preprocessor_config.json").write_text('{"merge_size":2}')
    (root / "model_index.json").write_text('{"_minimax_h3":{"partition":"fl2va"}}')
    return build_encoder_provenance(root)


def _bits(token_count: int = 3) -> np.ndarray:
    patterns = np.asarray([0x3F80, 0x4000, 0xC040, 0x0000], dtype=np.uint16)
    values = np.resize(patterns, token_count * CONDITIONING_WIDTH)
    return values.reshape(1, token_count, CONDITIONING_WIDTH)


def _write_fixture(root: Path, *, prompt: str = "a fixed prompt"):
    provenance = _checkpoint(root)
    path = root / "conditioning-artifact.npz"
    artifact = create_conditioning_artifact_from_bits(
        path,
        prompt=prompt,
        conditioning_bits=_bits(),
        text_token_tags=np.ones((3,), dtype=np.int32),
        token_ids=np.asarray([[11, 12, 13]], dtype=np.int32),
        encoder_provenance=provenance,
    )
    return path, artifact, provenance


class _LogicalBF16:
    dtype = "bfloat16"

    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.values, dtype=dtype)


class ConditioningArtifactContractTests(unittest.TestCase):
    def test_roundtrip_preserves_raw_bfloat16_shape_dtype_values_and_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, original, _provenance = _write_fixture(Path(directory))
            loaded = load_conditioning_artifact(path)

            self.assertEqual(loaded.conditioning_bits.dtype, np.dtype("uint16"))
            self.assertEqual(loaded.conditioning_bits.shape, (1, 3, CONDITIONING_WIDTH))
            self.assertTrue(np.array_equal(loaded.conditioning_bits, original.conditioning_bits))
            self.assertTrue(np.array_equal(loaded.text_token_tags, np.ones(3, dtype=np.int32)))
            self.assertTrue(np.array_equal(loaded.token_ids, np.asarray([[11, 12, 13]], dtype=np.int32)))
            self.assertEqual(loaded.metadata["conditioning"]["dtype"], "bfloat16")
            self.assertEqual(loaded.metadata["conditioning"]["tensor_checksum"], loaded.tensor_checksum)
            self.assertEqual(loaded.artifact_identity, original.artifact_identity)
            self.assertEqual(loaded.metadata["schema"]["version"], ARTIFACT_SCHEMA_VERSION)

            decoded = loaded.conditioning_float32()
            self.assertTrue(np.array_equal(decoded, bfloat16_bits_to_float32(_bits())))

    def test_float32_interchange_refuses_silent_bfloat16_narrowing(self) -> None:
        with self.assertRaisesRegex(ConditioningArtifactError, "refusing implicit narrowing"):
            bfloat16_bits_from_float32(np.asarray([1.1], dtype=np.float32))

    def test_logical_bfloat16_creation_preserves_exact_bits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provenance = _checkpoint(root)
            bits = _bits()
            artifact = create_conditioning_artifact(
                root / "logical-bfloat16.npz",
                prompt="logical bfloat16",
                conditioning=_LogicalBF16(bfloat16_bits_to_float32(bits)),
                text_token_tags=np.ones((3,), dtype=np.int32),
                token_ids=np.asarray([[11, 12, 13]], dtype=np.int32),
                encoder_provenance=provenance,
            )
            self.assertTrue(np.array_equal(artifact.conditioning_bits, bits))

    def test_prompt_and_h3_shape_compatibility_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, artifact, _provenance = _write_fixture(Path(directory))
            with self.assertRaisesRegex(ConditioningArtifactError, "prompt identity"):
                validate_conditioning_artifact(artifact, prompt="a different prompt")
            with self.assertRaisesRegex(ConditioningArtifactError, "text_dim"):
                validate_conditioning_artifact(artifact, text_dim=4096)
            self.assertEqual(load_conditioning_artifact(path).artifact_identity, artifact.artifact_identity)

    def test_wrong_encoder_identity_is_rejected_without_loading_qwen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, artifact, _provenance = _write_fixture(root)
            other = root / "other-checkpoint"
            _checkpoint(other)
            (other / "tokenizer" / "tokenizer_config.json").write_text('{"add_bos_token":true}')
            with self.assertRaisesRegex(ConditioningArtifactError, "identity"):
                validate_conditioning_artifact(artifact, checkpoint_root=other)
            (other / "tokenizer" / "tokenizer_config.json").write_text('{"add_bos_token":false}')
            (other / "processor" / "preprocessor_config.json").write_text('{"merge_size":4}')
            with self.assertRaisesRegex(ConditioningArtifactError, "identity"):
                validate_conditioning_artifact(artifact, checkpoint_root=other)
            (other / "processor" / "preprocessor_config.json").write_text('{"merge_size":2}')
            (other / "text_encoder" / "model.safetensors").write_bytes(b"changed encoder weights")
            with self.assertRaisesRegex(ConditioningArtifactError, "identity"):
                validate_conditioning_artifact(artifact, checkpoint_root=other)
            self.assertEqual(load_conditioning_artifact(path).artifact_identity, artifact.artifact_identity)

    def test_schema_corruption_tensor_corruption_and_missing_metadata_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, artifact, _provenance = _write_fixture(root)
            with np.load(path, allow_pickle=False) as loaded:
                arrays = {key: np.array(loaded[key], copy=True) for key in loaded.files}

            schema_path = root / "bad-schema.npz"
            metadata = json.loads(str(arrays["metadata_json"].item()))
            metadata["schema"]["version"] = 999
            arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
            np.savez_compressed(schema_path, **arrays)
            with self.assertRaisesRegex(ConditioningArtifactError, "schema version"):
                load_conditioning_artifact(schema_path)

            corrupt_path = root / "corrupt-tensor.npz"
            arrays = {key: np.array(value, copy=True) for key, value in arrays.items()}
            original_metadata = json.loads(str(arrays["metadata_json"].item()))
            original_metadata["schema"]["version"] = ARTIFACT_SCHEMA_VERSION
            original_metadata["artifact_identity"] = _artifact_identity(original_metadata)
            arrays["metadata_json"] = np.asarray(json.dumps(original_metadata, sort_keys=True, separators=(",", ":")))
            arrays["text_conditioning_bf16_bits"].flat[0] ^= np.uint16(1)
            np.savez_compressed(corrupt_path, **arrays)
            with self.assertRaisesRegex(ConditioningArtifactError, "checksum"):
                load_conditioning_artifact(corrupt_path)

            missing_path = root / "missing-metadata.npz"
            arrays = {key: np.array(value, copy=True) for key, value in arrays.items()}
            missing_metadata = json.loads(str(arrays["metadata_json"].item()))
            missing_metadata.pop("postprocessing")
            arrays["metadata_json"] = np.asarray(json.dumps(missing_metadata, sort_keys=True, separators=(",", ":")))
            np.savez_compressed(missing_path, **arrays)
            with self.assertRaisesRegex(ConditioningArtifactError, "missing required fields"):
                load_conditioning_artifact(missing_path)


if __name__ == "__main__":
    unittest.main()
