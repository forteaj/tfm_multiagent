from bug1 import is_admin


def test_is_admin_with_literal_string():
    assert is_admin("admin") is True


def test_is_admin_with_dynamically_created_string():
    role = "".join(["ad", "min"])

    assert is_admin(role) is True


def test_is_admin_with_other_role():
    assert is_admin("user") is False