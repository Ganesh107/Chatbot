import json

def load_products(json_path="app/products.json"):
    with open(json_path, "r") as f:
        products = json.load(f)
    return [p["description"] for p in products]
