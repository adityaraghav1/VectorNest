import pytest

from vectornest.core.exceptions import ValidationError
from vectornest.core.types import DistanceMetric
from vectornest.models.collection import CollectionConfig
    

def test_collection_config_normalizes_name_and_description() -> None:
    collection = CollectionConfig(
        name="  course_notes  ",
        dimension=768,
        distance_metric=DistanceMetric.COSINE,
        description="  Lecture embeddings  ",
    )

    assert collection.name == "course_notes"
    assert collection.description == "Lecture embeddings"


@pytest.mark.parametrize("dimension", [0, -1, True, 3.5])
def test_collection_config_rejects_invalid_dimension(dimension: object) -> None:
    with pytest.raises(ValidationError):
        CollectionConfig(name="notes", dimension=dimension) 


def test_collection_config_rejects_unsafe_name() -> None:
    with pytest.raises(ValidationError, match="only letters"):
        CollectionConfig(name="notes/2026", dimension=3)


def test_collection_config_rejects_non_string_text_fields() -> None:
    with pytest.raises(ValidationError):
        CollectionConfig(name=123, dimension=3)  
    with pytest.raises(ValidationError):
        CollectionConfig(name="notes", dimension=3, description=123)  

@pytest.mark.parametrize(
    "name",
    [
        "../secret",
        "..\\secret",
        "folder/collection",
        "folder\\collection",
        ".",
        "..",
        "/absolute",
        "\\absolute",
    ],
)
def test_collection_name_rejects_path_like_values(
    name: str,
) -> None:
    with pytest.raises(
        ValidationError,
        match=(
            "Collection name may contain only letters, "
            "digits, hyphens, and underscores."
        ),
    ):
        CollectionConfig(
            name=name,
            dimension=3,
        )


@pytest.mark.parametrize(
    "name",
    [
        "knowledge",
        "knowledge_base",
        "knowledge-base",
        "KnowledgeBase2026",
        "vector_nest-01",
    ],
)
def test_collection_name_accepts_safe_values(
    name: str,
) -> None:
    collection = CollectionConfig(
        name=name,
        dimension=3,
    )

    assert collection.name == name
