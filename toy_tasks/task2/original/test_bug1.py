from bug1 import reverse_items


def test_reverse_items_returns_reversed_list():
    items = [1, 2, 3]

    result = reverse_items(items)

    assert result == [3, 2, 1]


def test_reverse_items_does_not_modify_original_list():
    items = [1, 2, 3]

    reverse_items(items)

    assert items == [1, 2, 3]