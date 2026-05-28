# typed-datasets

A well-typed wrapper around [Hugging Face `datasets`](https://github.com/huggingface/datasets).

The `datasets` library returns rows as `dict[str, Any]`, which loses static type
information about columns. `typed-datasets` wraps `Dataset` and `IterableDataset`
in a generic container parameterized by a row type (typically a `TypedDict`), so
your IDE and type checker know what's in each row.

## Installation

```bash
pip install git+https://github.com/Plyb/typed-datasets.git
```

Or with uv:

```bash
uv add git+https://github.com/Plyb/typed-datasets.git
```

## Usage

```python
from typing import TypedDict
from typed_datasets import load_typed

class Row(TypedDict):
    text: str
    label: int

ds = load_typed("imdb", row_type=Row, split="train")

for row in ds.take(5):
    print(row["text"], row["label"])  # types known statically
```
