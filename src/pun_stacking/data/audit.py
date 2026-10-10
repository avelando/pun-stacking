from collections import Counter, defaultdict
import json
from pathlib import Path
import tomllib


SPLIT_NAMES = ("train", "validation", "test")

EXPECTED_COUNTS = {
    "train": 3990,
    "validation": 570,
    "test": 1140,
}

EXPECTED_LABEL_COUNTS = {
    "train": {"0": 1995, "1": 1995},
    "validation": {"0": 285, "1": 285},
    "test": {"0": 570, "1": 570},
}


def pair_id(example_id: str) -> str:
    if not isinstance(example_id, str) or "." not in example_id:
        raise ValueError(f"Invalid example ID: {example_id!r}")

    base, suffix = example_id.rsplit(".", 1)

    if not base or suffix not in {"H", "N"}:
        raise ValueError(f"Invalid example ID: {example_id!r}")

    return base


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset file: {path}")

    records = []

    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from error

            if not isinstance(record, dict):
                raise ValueError(f"Invalid record in {path}:{line_number}")

            example_id = record.get("id")
            pair_id(example_id)

            label = record.get("label")
            expected_label = 1 if example_id.endswith(".H") else 0

            if type(label) is not int or label != expected_label:
                raise ValueError(f"Invalid label for {example_id!r} in {path}")

            text = record.get("text")

            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Invalid text for {example_id!r} in {path}")

            if "tokens" in record or "labels" in record:
                tokens = record.get("tokens")
                labels = record.get("labels")

                if not isinstance(tokens, list) or not isinstance(labels, list):
                    raise ValueError(f"Invalid token annotations for {example_id!r}")

                if len(tokens) != len(labels):
                    raise ValueError(f"Inconsistent token annotations for {example_id!r}")

            records.append(record)

    return records


def inspect_splits(
    split_directory: Path,
    reference: dict[str, dict] | None = None,
    require_pairs: bool = False,
) -> dict:
    observed_ids = set()
    pair_splits = defaultdict(set)
    suffixes = defaultdict(set)

    counts = {}
    classes = {}

    for split_name in SPLIT_NAMES:
        path = split_directory / f"{split_name}.jsonl"
        rows = load_jsonl(path)

        counts[split_name] = len(rows)

        if len(rows) != EXPECTED_COUNTS[split_name]:
            raise ValueError(
                f"{path}: expected {EXPECTED_COUNTS[split_name]} rows, got {len(rows)}"
            )

        distribution = Counter(str(row["label"]) for row in rows)

        classes[split_name] = {label: distribution.get(label, 0) for label in ("0", "1")}

        if classes[split_name] != EXPECTED_LABEL_COUNTS[split_name]:
            raise ValueError(f"Unexpected class distribution: {path}")

        for row in rows:
            example_id = row["id"]

            if example_id in observed_ids:
                raise ValueError(f"Duplicate ID: {example_id}")

            observed_ids.add(example_id)

            if reference is not None:
                if example_id not in reference:
                    raise ValueError(f"Unknown example ID: {example_id}")

                if reference[example_id] != row:
                    raise ValueError(f"Record differs from original: {example_id}")

            group = pair_id(example_id)

            pair_splits[group].add(split_name)
            suffixes[group].add(example_id.rsplit(".", 1)[1])

    if len(observed_ids) != 5700 or len(pair_splits) != 2850:
        raise ValueError("Unexpected numbers of instances or pairs")

    if any(suffix_set != {"H", "N"} for suffix_set in suffixes.values()):
        raise ValueError("Every pair must include one .H and one .N")

    if reference is not None and observed_ids != reference.keys():
        raise ValueError("Reorganized split contains different example IDs")

    cross_count = sum(len(assigned_splits) > 1 for assigned_splits in pair_splits.values())

    if require_pairs and cross_count:
        raise ValueError(f"Found {cross_count} pairs crossing splits")

    return {
        "examples": len(observed_ids),
        "pairs": len(pair_splits),
        "cross_split_pairs": cross_count,
        "sizes": counts,
        "classes": classes,
    }


def audit_data(project_root: Path | None = None) -> Path:
    root = (project_root or Path.cwd()).resolve()
    config_path = root / "configs" / "experiments.toml"

    with config_path.open("rb") as stream:
        config = tomllib.load(stream)

    seeds = config["data"]["split_seeds"]

    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Split seeds must be unique and nonempty")

    data_root = root / config["data"].get("root", "data")
    original_dir = data_root / "original"

    canonical = {}

    for split_name in SPLIT_NAMES:
        path = original_dir / f"{split_name}.jsonl"

        for record in load_jsonl(path):
            example_id = record["id"]

            if example_id in canonical:
                raise ValueError(f"Duplicate original ID: {example_id}")

            canonical[example_id] = record

    source = inspect_splits(original_dir)

    partitions = []

    for seed in seeds:
        split_dir = data_root / "pair_controlled" / f"seed_{seed}"

        result = inspect_splits(
            split_dir,
            reference=canonical,
            require_pairs=True,
        )

        metadata_path = split_dir / "metadata.json"

        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing split metadata: {metadata_path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        if (
            metadata.get("strategy") != "pair_controlled"
            or metadata.get("seed") != seed
            or metadata.get("cross_split_pairs") != 0
            or metadata.get("total_examples") != result["examples"]
            or metadata.get("total_pairs") != result["pairs"]
            or metadata.get("class_distribution") != result["classes"]
        ):
            raise ValueError(f"Split metadata is inconsistent for seed {seed}")

        partitions.append(
            {
                "seed": seed,
                **result,
            }
        )

        print(f"Seed {seed}: {result['examples']} examples, {result['pairs']} pairs, 0 crossings")

    report = {
        "strategy": "pair_controlled",
        "source": source,
        "partitions": partitions,
    }

    output_path = root / "reports" / "data_integrity.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_suffix(".tmp")

    temporary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(output_path)

    print(f"Report: {output_path}")

    return output_path
