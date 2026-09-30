from __future__ import annotations

from warpaudit.cache.ledger import JobRecord, StatusLedger
from warpaudit.cache.store import ShardedTable


def test_interrupted_job_resumes_without_duplicate_rows(tmp_path) -> None:
    ledger = StatusLedger(tmp_path / "ledger.jsonl")
    ledger.append(JobRecord("job", "register", "running", attempt=1))
    with open(ledger.path, "a", encoding="utf-8") as fh:
        fh.write('{"torn":')
    ledger.append(JobRecord("job", "register", "done", status="ok", attempt=2))
    table = ShardedTable(tmp_path / "cache", "registrations")
    table.append([{"job_id": "job", "value": 1}], shard_hint="first")
    table.append([{"job_id": "job", "value": 2}], shard_hint="resume")
    assert ledger.completed("register") == {"job"}
    assert ledger.attempts("job") == 2
    assert table.count() == 1
    assert table.load().iloc[0]["value"] == 2
