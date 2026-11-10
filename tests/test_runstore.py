"""Round-trip the run store against an in-memory stand-in for R2.

No network and no boto3. The logic worth testing is the key layout, the
sha256 check and the refusal to overwrite a different file, none of which
needs a real bucket.
"""

from pathlib import Path

import pytest

from src.common import file_hash
from src.runstore import RunStore, credential, dotenv, secret


class NotFound(Exception):
    """Stands in for botocore's ClientError, which carries a `response` dict."""

    response = {"Error": {"Code": "404"}}


class FakeS3:
    """Just enough of the boto3 S3 client for RunStore."""

    def __init__(self):
        self.objects = {}

    def upload_file(self, path, bucket, key, ExtraArgs=None):
        data = Path(path).read_bytes()
        self.objects[key] = (data, (ExtraArgs or {}).get("Metadata", {}))

    def download_file(self, bucket, key, path):
        Path(path).write_bytes(self.objects[key][0])

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise NotFound(Key)
        data, meta = self.objects[Key]
        return {"ContentLength": len(data), "Metadata": meta}

    def get_paginator(self, _):
        return self

    def paginate(self, Bucket, Prefix, Delimiter=None):
        keys = [k for k in self.objects if k.startswith(Prefix)]
        if Delimiter is None:
            return [{"Contents": [{"Key": k} for k in sorted(keys)]}]
        folders = {Prefix + k[len(Prefix):].split(Delimiter)[0] + Delimiter for k in keys}
        return [{"CommonPrefixes": [{"Prefix": p} for p in sorted(folders)]}]


@pytest.fixture
def store():
    return RunStore(FakeS3(), "test-bucket")


@pytest.fixture
def run_folder(tmp_path):
    folder = tmp_path / "out"
    (folder / "sub").mkdir(parents=True)
    (folder / "results.csv").write_text("a,b\n1,2\n")
    (folder / "sub" / "metadata.json").write_text('{"run": "x"}')
    return folder


def test_round_trip_preserves_every_file(store, run_folder, tmp_path):
    store.put(run_folder, "nb02-run")
    back = store.get("nb02-run", tmp_path / "back")
    for original in run_folder.rglob("*"):
        if original.is_file():
            copy = back / original.relative_to(run_folder)
            assert copy.is_file()
            assert file_hash(copy) == file_hash(original)


def test_same_file_uploaded_twice_is_not_resent(store, run_folder):
    first = store.put(run_folder, "nb02-run")
    uploaded = dict(store.client.objects)
    assert store.put(run_folder, "nb02-run") == first
    assert store.client.objects == uploaded


def test_changed_file_needs_overwrite(store, run_folder):
    store.put(run_folder, "nb02-run")
    (run_folder / "results.csv").write_text("a,b\n9,9\n")
    with pytest.raises(FileExistsError):
        store.put(run_folder, "nb02-run")
    store.put(run_folder, "nb02-run", overwrite=True)


def test_corrupted_object_is_caught_on_download(store, run_folder, tmp_path):
    store.put(run_folder, "nb02-run")
    key = "dsr/nb02-run/results.csv"
    _, meta = store.client.objects[key]
    store.client.objects[key] = (b"truncated", meta)
    with pytest.raises(IOError):
        store.get("nb02-run", tmp_path / "back")


def test_missing_run_is_an_error_not_an_empty_folder(store, tmp_path):
    with pytest.raises(FileNotFoundError):
        store.get("never-ran", tmp_path / "back")


def test_runs_lists_labels(store, run_folder):
    store.put(run_folder, "nb01-checks")
    store.put(run_folder, "nb02-run")
    assert store.runs() == ["nb01-checks", "nb02-run"]


@pytest.mark.parametrize("label", ["", "a/b", "..", "."])
def test_bad_run_labels_are_refused(store, run_folder, label):
    with pytest.raises(ValueError):
        store.put(run_folder, label)


def test_dotenv_reads_pairs_and_skips_noise(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        "# a comment\n"
        "\n"
        "R2_ENDPOINT=https://acct.r2.cloudflarestorage.com\n"
        '  R2_ACCESS_KEY_ID = "spaced"  \n'
        "R2_SECRET_ACCESS_KEY='has=equals=inside'\n"
        "not a pair\n",
        encoding="utf-8")
    assert dotenv(path) == {
        "R2_ENDPOINT": "https://acct.r2.cloudflarestorage.com",
        "R2_ACCESS_KEY_ID": "spaced",
        "R2_SECRET_ACCESS_KEY": "has=equals=inside",
    }


def test_dotenv_without_a_file_is_empty(tmp_path):
    assert dotenv(tmp_path / "absent") == {}


def test_secret_prefers_the_environment_over_the_file(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("R2_ENDPOINT=from-file\n", encoding="utf-8")
    monkeypatch.setattr("src.runstore.ENV_FILE", path)
    monkeypatch.setenv("R2_ENDPOINT", "from-env")
    assert secret("R2_ENDPOINT") == "from-env"
    monkeypatch.delenv("R2_ENDPOINT")
    assert secret("R2_ENDPOINT") == "from-file"


def test_project_prefix_wins_over_the_old_name(monkeypatch, tmp_path):
    monkeypatch.setattr("src.runstore.ENV_FILE", tmp_path / "absent")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "hypernetwork-token")
    assert credential("ACCESS_KEY_ID") == "hypernetwork-token"
    monkeypatch.setenv("DSR_ACCESS_KEY_ID", "this-project")
    assert credential("ACCESS_KEY_ID") == "this-project"
