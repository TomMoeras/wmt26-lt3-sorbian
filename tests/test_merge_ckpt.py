import subprocess
import sys

from lt3wmt26.merge_ckpt import EXPECTED_EOS


def test_expected_eos_is_the_documented_list():
    assert EXPECTED_EOS == [248046, 248044]


def test_base_is_required_no_default():
    out = subprocess.run([sys.executable, "-m", "lt3wmt26.merge_ckpt",
                          "--ckpt", "/x", "--out", "/y"],
                         capture_output=True, text=True)
    assert out.returncode != 0
    assert "--base" in out.stderr


def test_module_imports_without_peft_or_gpu_deps():
    # peft/torch/transformers are only imported inside merge(); importing the module itself
    # (and building its argparser) must not require them.
    out = subprocess.run([sys.executable, "-c", "import lt3wmt26.merge_ckpt"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
