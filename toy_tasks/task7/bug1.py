def add_item(item, items=None):
    if items is None:
        # When no list is provided, return a new list containing only the item.
        return [item]
    else:
        # If a list is provided, modify it in place and return it.
        items.append(item)
        return items