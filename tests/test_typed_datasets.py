from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Literal, TypedDict

import datasets
import pytest

from typed_datasets import (
    TypedDataset,
    TypedDatasetDict,
    TypedIterableDataset,
    TypedIterableDatasetDict,
    concatenate,
    interleave,
)


class Row(TypedDict):
    name: str
    age: int


class UpperRow(TypedDict):
    name_upper: str
    doubled_age: int


def make_dataset() -> TypedDataset[Row]:
    return TypedDataset.from_list([
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
        {"name": "Carol", "age": 40},
        {"name": "Dave", "age": 35},
    ])


def test_from_list_roundtrip():
    ds = make_dataset()
    assert len(ds) == 4
    assert ds[0] == {"name": "Alice", "age": 30}
    assert list(iter(ds)) == [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
        {"name": "Carol", "age": 40},
        {"name": "Dave", "age": 35},
    ]


def test_from_iterable():
    rows: list[Row] = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
    ds = TypedDataset.from_iterable(iter(rows))
    assert len(ds) == 2


def test_from_generator():
    def gen() -> Iterator[Row]:
        for i in range(3):
            yield {"name": f"X{i}", "age": i}

    ds = TypedDataset.from_generator(gen)
    assert len(ds) == 3
    assert ds[2] == {"name": "X2", "age": 2}


def test_columns_and_features():
    ds = make_dataset()
    assert ds.columns == ("name", "age")
    assert "name" in ds.features
    assert "age" in ds.features


def test_repr_passthrough():
    ds = make_dataset()
    text = repr(ds)
    assert "num_rows" in text
    assert "4" in text


def test_map_changes_row_type():
    ds = make_dataset()

    def upper(row: Row) -> UpperRow:
        return {"name_upper": row["name"].upper(), "doubled_age": row["age"] * 2}

    mapped = ds.map(upper, remove_columns=["name", "age"])
    assert mapped.columns == ("name_upper", "doubled_age")
    assert mapped[0] == {"name_upper": "ALICE", "doubled_age": 60}


def test_filter():
    ds = make_dataset()
    older = ds.filter(lambda r: r["age"] >= 30)
    assert len(older) == 3
    assert all(r["age"] >= 30 for r in older)


def test_select_skip_take_shard():
    ds = make_dataset()
    assert len(ds.select([0, 2])) == 2
    assert ds.skip(2)[0] == {"name": "Carol", "age": 40}
    assert ds.take(2)[1] == {"name": "Bob", "age": 25}
    assert len(ds.shard(2, 0)) == 2


def test_flatten_indices():
    ds = make_dataset().select([3, 1, 0]).flatten_indices()
    assert [r["name"] for r in ds] == ["Dave", "Bob", "Alice"]


def test_concatenate():
    a = make_dataset()
    b: TypedDataset[Row] = TypedDataset.from_list([{"name": "Eve", "age": 50}])
    c = concatenate([a, b])
    assert len(c) == 5
    assert c[4] == {"name": "Eve", "age": 50}


def test_interleave():
    a: TypedDataset[Row] = TypedDataset.from_list([{"name": "A1", "age": 1}, {"name": "A2", "age": 2}])
    b: TypedDataset[Row] = TypedDataset.from_list([{"name": "B1", "age": 10}, {"name": "B2", "age": 20}])
    merged = interleave([a, b], seed=0)
    names = [r["name"] for r in merged]
    assert set(names) <= {"A1", "A2", "B1", "B2"}


def test_save_load_roundtrip(tmp_path: Path):
    ds = make_dataset()
    ds.save_to_disk(tmp_path / "saved")
    loaded = TypedDataset.load_from_disk(tmp_path / "saved", row_type=Row)
    assert list(loaded) == list(ds)


def test_train_test_split_dict_keys():
    ds = make_dataset()
    split = ds.train_test_split(test_size=0.5, seed=0)
    assert isinstance(split, TypedDatasetDict)
    assert "train" in split
    assert "test" in split
    assert len(split["train"]) + len(split["test"]) == 4


def test_dataset_dict_iteration_and_map():
    ds = make_dataset()
    split = ds.train_test_split(test_size=0.5, seed=0)
    assert sorted(split.keys()) == ["test", "train"]
    sizes = {k: len(v) for k, v in split.items()}
    assert sum(sizes.values()) == 4

    def lower_name(r: Row) -> Row:
        return {"name": r["name"].lower(), "age": r["age"]}

    mapped = split.map(lower_name)
    for ds_ in mapped.values():
        for row in ds_:
            assert row["name"] == row["name"].lower()


def test_dataset_dict_save_load(tmp_path: Path):
    ds = make_dataset()
    split = ds.train_test_split(test_size=0.5, seed=0)
    split.save_to_disk(tmp_path / "dd")
    loaded = TypedDatasetDict.load_from_disk(tmp_path / "dd", key_type=str, row_type=Row)
    assert set(loaded.keys()) == {"train", "test"}


def test_unwrap_round_trip():
    ds = make_dataset()
    raw = ds.unwrap()
    assert isinstance(raw, datasets.Dataset)
    rewrapped: TypedDataset[Row] = TypedDataset(raw)
    assert list(rewrapped) == list(ds)


def test_iterable_basics():
    ds = make_dataset().to_iterable_dataset()
    assert isinstance(ds, TypedIterableDataset)
    rows = list(ds.take(2))
    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"


def test_iterable_dataset_dict_construct():
    base = make_dataset()
    src = datasets.IterableDatasetDict({
        "train": base.unwrap().to_iterable_dataset(),
        "validation": base.unwrap().to_iterable_dataset(),
    })
    wrapped: TypedIterableDatasetDict[Literal["train", "validation"], Row] = TypedIterableDatasetDict(src)
    assert set(wrapped.keys()) == {"train", "validation"}
    train_rows = list(wrapped["train"].take(1))
    assert train_rows[0]["name"] == "Alice"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
