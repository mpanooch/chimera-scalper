# data_converter/bybit_converter.py

import json
import gzip
from pathlib import Path

class BybitConverter:
    """
    Converts Bybit raw order book data (from REST or WebSocket dumps)
    into:
      - snapshot.json
      - updates.jsonl
    """

    def __init__(self, input_file, out_dir="data"):
        self.input_file = input_file
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True)

    def convert(self):
        snapshot = None
        updates = []

        # auto-detect if file is gzipped
        opener = gzip.open if self.input_file.endswith(".gz") else open

        with opener(self.input_file, "rt") as f:
            for line in f:
                try:
                    msg = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                # Bybit snapshot (REST response or websocket "snapshot")
                if "type" in msg and msg["type"] == "snapshot":
                    snapshot = {"b": msg["b"], "a": msg["a"]}

                # Bybit delta (websocket "delta")
                elif "type" in msg and msg["type"] == "delta":
                    updates.append({"b": msg.get("b", []), "a": msg.get("a", [])})

        if snapshot is None:
            raise ValueError("No snapshot found in input file!")

        # Write snapshot.json
        snapshot_file = self.out_dir / "snapshot.json"
        with open(snapshot_file, "w") as f:
            json.dump(snapshot, f)

        # Write updates.jsonl
        updates_file = self.out_dir / "updates.jsonl"
        with open(updates_file, "w") as f:
            for u in updates:
                f.write(json.dumps(u) + "\n")

        print(f"✅ Conversion complete")
        print(f"Snapshot saved: {snapshot_file}")
        print(f"Updates saved:  {updates_file}")
