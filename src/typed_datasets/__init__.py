from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from os import PathLike
from typing import Any, Literal, overload

import datasets


class TypedDataset[RowT: Mapping[str, Any]]:
    def __init__(self, dataset: datasets.Dataset) -> None:
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __iter__(self) -> Iterator[RowT]:
        return iter(self._dataset)  # type: ignore[return-value]

    def __contains__(self, item: object) -> bool:
        return item in self._dataset

    def __getitem__(self, index: int) -> RowT:
        return self._dataset[index]  # type: ignore[return-value]

    def __repr__(self) -> str:
        return repr(self._dataset)

    def unwrap(self) -> datasets.Dataset:
        return self._dataset

    @property
    def features(self) -> datasets.Features:
        return self._dataset.features

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self._dataset.column_names)

    def skip(self, n: int) -> TypedDataset[RowT]:
        # Note: we don't do `TypedDataset[RowT]` here because dill trips on it when trying to fingerprint the dataset.
        # It also doesn't actually buy us any static type safety, since the constructor doesn't mention `RowT` (there isn't
        # anything to check against to see if the TypedDataset is being constructed correctly).
        # Python's runtime semantics mean that `TypedDataset[RowT]` only references the `TypeVar` `RowT`, not the actual
        # concrete type of the instance on which this (and other methods) are called.
        return TypedDataset(self._dataset.select(range(n, len(self._dataset))))

    def take(self, n: int) -> TypedDataset[RowT]:
        return TypedDataset(self._dataset.select(range(n)))

    def select(self, indices: Iterable[int]) -> TypedDataset[RowT]:
        return TypedDataset(self._dataset.select(indices))

    def shard(self, num_shards: int, index: int, contiguous: bool = False) -> TypedDataset[RowT]:
        return TypedDataset(self._dataset.shard(num_shards, index, contiguous=contiguous))

    def flatten_indices(self) -> TypedDataset[RowT]:
        return TypedDataset(self._dataset.flatten_indices())

    @overload
    def map[OutRow: Mapping[str, Any]](self, fn: Callable[[RowT], OutRow], *, with_indices: Literal[False] = ..., **kwargs: Any) -> TypedDataset[OutRow]: ...
    @overload
    def map[OutRow: Mapping[str, Any]](self, fn: Callable[[RowT, int], OutRow], *, with_indices: Literal[True], **kwargs: Any) -> TypedDataset[OutRow]: ...
    def map[OutRow: Mapping[str, Any]](self, fn: Callable[..., OutRow], *, with_indices: bool = False, **kwargs: Any) -> TypedDataset[OutRow]:
        return TypedDataset(self._dataset.map(fn, with_indices=with_indices, **kwargs))

    def filter(self, fn: Callable[[RowT], bool], **kwargs: Any) -> TypedDataset[RowT]:
        return TypedDataset(self._dataset.filter(fn, **kwargs))

    def shuffle(self, seed: int | None = None) -> TypedDataset[RowT]:
        return TypedDataset(self._dataset.shuffle(seed=seed))

    def train_test_split(
        self,
        test_size: float | int | None = None,
        train_size: float | int | None = None,
        seed: int | None = None,
    ) -> TypedDatasetDict[Literal["train", "test"], RowT]:
        return TypedDatasetDict(
            self._dataset.train_test_split(test_size=test_size, train_size=train_size, seed=seed)
        )

    def to_iterable_dataset(self, num_shards: int | None = 1) -> TypedIterableDataset[RowT]:
        return TypedIterableDataset(self._dataset.to_iterable_dataset(num_shards=num_shards))

    def save_to_disk(
        self,
        dataset_path: str | PathLike[str],
        max_shard_size: str | int | None = None,
        num_shards: int | None = None,
        num_proc: int | None = None,
        storage_options: dict[str, Any] | None = None,
    ) -> None:
        self._dataset.save_to_disk(dataset_path, max_shard_size, num_shards, num_proc, storage_options)

    @classmethod
    def from_list[R: Mapping[str, Any]](cls, rows: list[R]) -> TypedDataset[R]:
        return TypedDataset(datasets.Dataset.from_list(rows))  # pyright: ignore[reportArgumentType]

    @classmethod
    def from_iterable[R: Mapping[str, Any]](cls, rows: Iterable[R]) -> TypedDataset[R]:
        return TypedDataset(datasets.Dataset.from_list(list(rows)))  # pyright: ignore[reportArgumentType]

    @classmethod
    def from_generator[R: Mapping[str, Any]](
        cls,
        generator: Callable[..., Iterable[R]],
        **kwargs: Any,
    ) -> TypedDataset[R]:
        result = datasets.Dataset.from_generator(generator, **kwargs)
        if not isinstance(result, datasets.Dataset):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"from_generator returned {type(result).__name__}; expected Dataset")
        return TypedDataset(result)

    @classmethod
    def load_from_disk[R: Mapping[str, Any]](
        cls,
        path: str | PathLike[str],
        *,
        row_type: type[R],
    ) -> TypedDataset[R]:
        raw = datasets.Dataset.load_from_disk(path)
        if not isinstance(raw, datasets.Dataset):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"Path {path!r} loaded as {type(raw).__name__}; expected Dataset")
        return TypedDataset(raw)


class TypedIterableDataset[RowT: Mapping[str, Any]]:
    def __init__(self, dataset: datasets.IterableDataset) -> None:
        self._dataset = dataset

    def __iter__(self) -> Iterator[RowT]:
        return iter(self._dataset)  # type: ignore[return-value]

    def __repr__(self) -> str:
        return repr(self._dataset)

    def unwrap(self) -> datasets.IterableDataset:
        return self._dataset

    @property
    def features(self) -> datasets.Features | None:
        return self._dataset.features

    @property
    def columns(self) -> tuple[str, ...] | None:
        cols = self._dataset.column_names
        return tuple(cols) if cols is not None else None

    def skip(self, n: int) -> TypedIterableDataset[RowT]:
        return TypedIterableDataset(self._dataset.skip(n))

    def take(self, n: int) -> TypedIterableDataset[RowT]:
        return TypedIterableDataset(self._dataset.take(n))

    @overload
    def map[OutRow: Mapping[str, Any]](self, fn: Callable[[RowT], OutRow], *, with_indices: Literal[False] = ..., **kwargs: Any) -> TypedIterableDataset[OutRow]: ...
    @overload
    def map[OutRow: Mapping[str, Any]](self, fn: Callable[[RowT, int], OutRow], *, with_indices: Literal[True], **kwargs: Any) -> TypedIterableDataset[OutRow]: ...
    def map[OutRow: Mapping[str, Any]](self, fn: Callable[..., OutRow], *, with_indices: bool = False, **kwargs: Any) -> TypedIterableDataset[OutRow]:
        return TypedIterableDataset(self._dataset.map(fn, with_indices=with_indices, **kwargs))

    def filter(self, fn: Callable[[RowT], bool], **kwargs: Any) -> TypedIterableDataset[RowT]:
        return TypedIterableDataset(self._dataset.filter(fn, **kwargs))

    def shuffle(self, seed: int | None = None, buffer_size: int = 1000) -> TypedIterableDataset[RowT]:
        return TypedIterableDataset(self._dataset.shuffle(seed=seed, buffer_size=buffer_size))


class TypedDatasetDict[K: str, RowT: Mapping[str, Any]]:
    def __init__(self, dataset_dict: datasets.DatasetDict) -> None:
        self._dataset_dict = dataset_dict

    def __len__(self) -> int:
        return len(self._dataset_dict)

    def __iter__(self) -> Iterator[K]:
        return iter(self._dataset_dict)  # type: ignore[return-value]

    def __contains__(self, key: object) -> bool:
        return key in self._dataset_dict

    def __getitem__(self, key: K) -> TypedDataset[RowT]:
        return TypedDataset(self._dataset_dict[key])

    def __repr__(self) -> str:
        return repr(self._dataset_dict)

    def unwrap(self) -> datasets.DatasetDict:
        return self._dataset_dict

    def keys(self) -> Iterable[K]:
        return self._dataset_dict.keys()  # type: ignore[return-value]

    def values(self) -> Iterable[TypedDataset[RowT]]:
        return (TypedDataset(d) for d in self._dataset_dict.values())

    def items(self) -> Iterable[tuple[K, TypedDataset[RowT]]]:
        return ((k, TypedDataset(d)) for k, d in self._dataset_dict.items())  # type: ignore[misc]

    @overload
    def map[OutRow: Mapping[str, Any]](self, fn: Callable[[RowT], OutRow], *, with_indices: Literal[False] = ..., **kwargs: Any) -> TypedDatasetDict[K, OutRow]: ...
    @overload
    def map[OutRow: Mapping[str, Any]](self, fn: Callable[[RowT, int], OutRow], *, with_indices: Literal[True], **kwargs: Any) -> TypedDatasetDict[K, OutRow]: ...
    def map[OutRow: Mapping[str, Any]](self, fn: Callable[..., OutRow], *, with_indices: bool = False, **kwargs: Any) -> TypedDatasetDict[K, OutRow]:
        return TypedDatasetDict(self._dataset_dict.map(fn, with_indices=with_indices, **kwargs))

    def filter(self, fn: Callable[[RowT], bool], **kwargs: Any) -> TypedDatasetDict[K, RowT]:
        return TypedDatasetDict(self._dataset_dict.filter(fn, **kwargs))

    def shuffle(self, seed: int | None = None) -> TypedDatasetDict[K, RowT]:
        return TypedDatasetDict(self._dataset_dict.shuffle(seeds=seed))

    def save_to_disk(
        self,
        dataset_dict_path: str | PathLike[str],
        max_shard_size: str | int | None = None,
        num_shards: dict[str, int] | None = None,
        num_proc: int | None = None,
        storage_options: dict[str, Any] | None = None,
    ) -> None:
        self._dataset_dict.save_to_disk(
            dataset_dict_path, max_shard_size, num_shards, num_proc, storage_options
        )

    @classmethod
    def load_from_disk[Kk: str, R: Mapping[str, Any]](
        cls,
        path: str | PathLike[str],
        *,
        key_type: type[Kk],
        row_type: type[R],
    ) -> TypedDatasetDict[Kk, R]:
        raw = datasets.DatasetDict.load_from_disk(path)
        if not isinstance(raw, datasets.DatasetDict):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"Path {path!r} loaded as {type(raw).__name__}; expected DatasetDict")
        return TypedDatasetDict(raw)


class TypedIterableDatasetDict[K: str, RowT: Mapping[str, Any]]:
    def __init__(self, dataset_dict: datasets.IterableDatasetDict) -> None:
        self._dataset_dict = dataset_dict

    def __iter__(self) -> Iterator[K]:
        return iter(self._dataset_dict)  # type: ignore[return-value]

    def __contains__(self, key: object) -> bool:
        return key in self._dataset_dict

    def __getitem__(self, key: K) -> TypedIterableDataset[RowT]:
        return TypedIterableDataset(self._dataset_dict[key])

    def __repr__(self) -> str:
        return repr(self._dataset_dict)

    def unwrap(self) -> datasets.IterableDatasetDict:
        return self._dataset_dict

    def keys(self) -> Iterable[K]:
        return self._dataset_dict.keys()  # type: ignore[return-value]

    def values(self) -> Iterable[TypedIterableDataset[RowT]]:
        return (TypedIterableDataset(d) for d in self._dataset_dict.values())

    def items(self) -> Iterable[tuple[K, TypedIterableDataset[RowT]]]:
        return ((k, TypedIterableDataset(d)) for k, d in self._dataset_dict.items())  # type: ignore[misc]

    @overload
    def map[OutRow: Mapping[str, Any]](self, fn: Callable[[RowT], OutRow], *, with_indices: Literal[False] = ..., **kwargs: Any) -> TypedIterableDatasetDict[K, OutRow]: ...
    @overload
    def map[OutRow: Mapping[str, Any]](self, fn: Callable[[RowT, int], OutRow], *, with_indices: Literal[True], **kwargs: Any) -> TypedIterableDatasetDict[K, OutRow]: ...
    def map[OutRow: Mapping[str, Any]](self, fn: Callable[..., OutRow], *, with_indices: bool = False, **kwargs: Any) -> TypedIterableDatasetDict[K, OutRow]:
        return TypedIterableDatasetDict(self._dataset_dict.map(fn, with_indices=with_indices, **kwargs))

    def filter(self, fn: Callable[[RowT], bool], **kwargs: Any) -> TypedIterableDatasetDict[K, RowT]:
        return TypedIterableDatasetDict(self._dataset_dict.filter(fn, **kwargs))

    def shuffle(self, seed: int | None = None, buffer_size: int = 1000) -> TypedIterableDatasetDict[K, RowT]:
        return TypedIterableDatasetDict(self._dataset_dict.shuffle(seed=seed, buffer_size=buffer_size))


def concatenate[R: Mapping[str, Any]](parts: Sequence[TypedDataset[R]]) -> TypedDataset[R]:
    return TypedDataset(datasets.concatenate_datasets([p.unwrap() for p in parts]))


def interleave[R: Mapping[str, Any]](
    parts: Sequence[TypedDataset[R]],
    probabilities: list[float] | None = None,
    seed: int | None = None,
    stopping_strategy: Literal["first_exhausted", "all_exhausted"] = "first_exhausted",
) -> TypedDataset[R]:
    combined = datasets.interleave_datasets(
        [p.unwrap() for p in parts],
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )
    if not isinstance(combined, datasets.Dataset):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"interleave_datasets returned {type(combined).__name__}; expected Dataset")
    return TypedDataset(combined)


@overload
def load_typed[R: Mapping[str, Any]](
    path: str,
    *,
    row_type: type[R],
    split: str,
    streaming: Literal[False] = ...,
    **kwargs: Any,
) -> TypedDataset[R]: ...
@overload
def load_typed[R: Mapping[str, Any]](
    path: str,
    *,
    row_type: type[R],
    split: str,
    streaming: Literal[True],
    **kwargs: Any,
) -> TypedIterableDataset[R]: ...
def load_typed[R: Mapping[str, Any]](
    path: str,
    *,
    row_type: type[R],
    split: str,
    streaming: bool = False,
    **kwargs: Any,
) -> TypedDataset[R] | TypedIterableDataset[R]:
    result = datasets.load_dataset(path, split=split, streaming=streaming, **kwargs)
    if streaming:
        if not isinstance(result, datasets.IterableDataset):
            raise TypeError(f"Expected IterableDataset for streaming, got {type(result).__name__}")
        return TypedIterableDataset(result)
    if not isinstance(result, datasets.Dataset):
        raise TypeError(f"Expected Dataset, got {type(result).__name__}")
    return TypedDataset(result)
