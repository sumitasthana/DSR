"""Keep notebook run outputs in Cloudflare R2 so a Colab session is not the only copy.

`outputs/` is gitignored and Colab's disk is wiped when the runtime ends, so a
run that writes only locally is lost. This pushes the whole output folder to R2
under its run label, and pulls it back for later notebooks and for writing up
the trajectory record.

The notebook side is two lines:

    store = connect()                 # None when no credentials are present
    store and store.put(OUT, RUN)

Only small files belong here: CSVs, PNGs, JSON. Model checkpoints do not. There
is no resume and no partial upload, because nothing we save takes long enough
to need one.
"""

from pathlib import Path, PurePosixPath
import os
import re

from .common import file_hash

#: Everything this project writes lives under one prefix, so the bucket can be
#: shared with other work without the two colliding.
PREFIX = "dsr"
BUCKET = "dsr-runs"   # This project only. The hypernetwork work uses its own bucket.


#: A local .env, for runs outside Colab. Colab secrets still win over it, so a
#: stale file in a clone cannot shadow the values set in the session.
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def dotenv(path=None):
    """`KEY=value` lines from a .env file. Empty when there is no file."""
    path = path or ENV_FILE
    values = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def secret(name):
    """An environment variable, then Colab's secret store, then a local .env."""
    value = os.environ.get(name)
    if not value:
        try:
            from google.colab import userdata
            value = userdata.get(name)
        except Exception:
            value = None
    if not value:
        value = dotenv().get(name)
    return value


#: One Colab account holds credentials for several projects, so this project's
#: secrets carry its own prefix. `R2_*` is the older unprefixed naming, still
#: accepted so an existing local .env or shell keeps working.
FIELDS = ("ENDPOINT", "ACCESS_KEY_ID", "SECRET_ACCESS_KEY")


def credential(field):
    """One R2 credential, preferring this project's `DSR_` name."""
    return secret(f"DSR_{field}") or secret(f"R2_{field}")


def connect(bucket=BUCKET, *, required=False):
    """A store, or None when credentials are missing.

    Returning None rather than raising lets a notebook run end to end without
    R2 configured. `required=True` turns a missing secret into an error, for
    the case where losing the run is the worse outcome.
    """
    parts = [credential(n) for n in FIELDS]
    if not all(parts):
        if required:
            raise RuntimeError(
                "Set DSR_ENDPOINT, DSR_ACCESS_KEY_ID and DSR_SECRET_ACCESS_KEY, "
                "as environment variables or as Colab secrets.")
        return None
    endpoint, key_id, key_secret = parts
    if not re.fullmatch(r"https://[a-z0-9.]+\.r2\.cloudflarestorage\.com", endpoint):
        raise ValueError("DSR_ENDPOINT must be the HTTPS S3 API endpoint from the R2 dashboard.")
    import boto3
    from botocore.config import Config
    client = boto3.client(
        "s3", endpoint_url=endpoint, region_name="auto",
        aws_access_key_id=key_id, aws_secret_access_key=key_secret,
        config=Config(signature_version="s3v4", connect_timeout=10, read_timeout=60,
                      retries={"mode": "standard", "max_attempts": 3}))
    return RunStore(client, bucket)


class RunStore:
    """Upload and download one run's output folder."""

    def __init__(self, client, bucket, prefix=PREFIX):
        self.client = client
        self.bucket = bucket
        self.prefix = prefix

    def _key(self, run, name=""):
        if not run or "/" in run or run in (".", ".."):
            raise ValueError(f"Run label must be a single path segment: {run!r}")
        return f"{self.prefix}/{run}" + (f"/{name}" if name else "")

    def _head(self, key):
        try:
            return self.client.head_object(Bucket=self.bucket, Key=key)
        except Exception as error:
            code = getattr(error, "response", {}).get("Error", {}).get("Code")
            if code in ("404", "NoSuchKey", "NotFound"):
                return None
            raise

    def put(self, folder, run, *, overwrite=False):
        """Upload every file in `folder` under `run`. Returns the keys written.

        Each object carries its sha256, and the upload is checked against the
        object that lands. Without that check a truncated upload looks like a
        saved run until someone tries to read it months later.
        """
        folder = Path(folder)
        written = []
        for path in sorted(p for p in folder.rglob("*") if p.is_file()):
            name = path.relative_to(folder).as_posix()
            key = self._key(run, name)
            digest, size = file_hash(path), path.stat().st_size
            there = self._head(key)
            if there is not None:
                same = (there.get("Metadata", {}).get("sha256") == digest
                        and there["ContentLength"] == size)
                if same:
                    written.append(key)
                    continue
                if not overwrite:
                    raise FileExistsError(
                        f"R2 already holds a different {key}. Use a new run label, "
                        "or pass overwrite=True if you mean to replace it.")
            self.client.upload_file(str(path), self.bucket, key,
                                    ExtraArgs={"Metadata": {"sha256": digest}})
            landed = self._head(key)
            if landed is None or landed["ContentLength"] != size:
                raise IOError(f"Upload did not land intact: {key}")
            written.append(key)
        return written

    def get(self, run, dest):
        """Download one run into `dest`, checking each file against its sha256."""
        dest = Path(dest)
        prefix = self._key(run) + "/"
        pulled = []
        for key in self.keys(prefix):
            path = dest / PurePosixPath(key).relative_to(prefix)
            path.parent.mkdir(parents=True, exist_ok=True)
            digest = (self._head(key) or {}).get("Metadata", {}).get("sha256", "")
            if path.is_file() and digest and file_hash(path) == digest:
                pulled.append(path)
                continue
            self.client.download_file(self.bucket, key, str(path))
            if digest and file_hash(path) != digest:
                raise IOError(f"Download does not match its recorded sha256: {key}")
            pulled.append(path)
        if not pulled:
            raise FileNotFoundError(f"No run named {run!r} in R2 under {self.prefix}/.")
        return dest

    def keys(self, prefix):
        pages = self.client.get_paginator("list_objects_v2").paginate(
            Bucket=self.bucket, Prefix=prefix)
        for page in pages:
            for item in page.get("Contents", []):
                yield item["Key"]

    def runs(self):
        """Every run label present, newest-looking last."""
        pages = self.client.get_paginator("list_objects_v2").paginate(
            Bucket=self.bucket, Prefix=self.prefix + "/", Delimiter="/")
        found = [item["Prefix"].rstrip("/").rsplit("/", 1)[-1]
                 for page in pages for item in page.get("CommonPrefixes", [])]
        return sorted(found)
