from typing import TypedDict

class User(TypedDict):
    id: int
    name: str
    email: str

new_user: User = {
    "id": 1,
    "name": "Alice",
    "email": "alice@example.com"
}

print(new_user)