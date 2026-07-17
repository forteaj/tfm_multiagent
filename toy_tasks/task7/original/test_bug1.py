from bug1 import add_item


def test_add_item_returns_new_list_each_time():
    first = add_item("a")
    second = add_item("b")

    assert first == ["a"]
    assert second == ["b"]


def test_add_item_uses_existing_list_when_provided():
    items = ["x"]

    result = add_item("y", items)

    assert result == ["x", "y"]
    assert items == ["x", "y"]