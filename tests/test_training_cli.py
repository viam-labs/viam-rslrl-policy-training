"""The contract with Viam's launcher: the args it passes, and strict parsing of ours."""

from pathlib import Path

import pytest

from model.training import parse_args


def test_viam_args_with_placeholder_dataset_and_quoted_custom_args():
    args = parse_args(
        [
            "--model_output_directory=/gcs/bucket/out",
            "--dataset_file=/gcs/bucket/dataset.jsonl",  # ignored: training is in simulation
            "--clearance_mm='1.0'",  # Viam can leave the shell quotes on
            '--run_name="first-run"',
            "--upload_part_id=abc",
        ]
    )
    assert args.model_output_directory == Path("/gcs/bucket/out")
    assert args.clearance_mm == 1.0
    assert args.run_name == "first-run"
    assert args.upload_part_id == "abc"
    assert args.mode == "train"


def test_sequence_dataset_args_are_accepted_too():
    args = parse_args(
        [
            "--model_output_directory=/o",
            "--binary_data_file=/b",
            "--tabular_data_file=/t",
            "--sequences_file=/s",
        ]
    )
    assert args.run_name.startswith("peg-insert-")


@pytest.mark.parametrize("bad", ["--bogus=1", "--max_iter=5"])  # unknown, and an abbreviation
def test_unknown_or_abbreviated_args_exit_2(bad):
    with pytest.raises(SystemExit) as e:
        parse_args(["--model_output_directory=/o", bad])
    assert e.value.code == 2
