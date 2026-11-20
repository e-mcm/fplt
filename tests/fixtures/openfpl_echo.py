"""Stand-in OpenFPL runner. Scores each row by the length of its player id."""

from __future__ import annotations

import json
import sys


def main() -> None:
    """Rewrite stdin rows. The mode is the first argument."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "length"
    payload = json.load(sys.stdin)
    if mode == "badjson":
        sys.stdout.write("nope")
        return
    rows = []
    for row in payload["rows"]:
        if mode == "drop":
            continue
        name = row["openfpl_player_id"]
        if mode == "unknown":
            name = "Nobody"
        elif mode == "rename":
            name = "Saka"
        rows.append(
            {
                "row_id": row["row_id"],
                "season": row["season"],
                "openfpl_player_id": name,
                "prediction": float(len(row["openfpl_player_id"])),
            }
        )
    json.dump({"rows": rows}, sys.stdout)


if __name__ == "__main__":
    main()
