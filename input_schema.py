INPUT_SCHEMA = {
    "text": {
        "example": [
           "Apple made the iphone"
        ],
        "shape": [1],
        "datatype": "STRING",
        "required": True,
    },
    "competitors": {
        "example": ["Apple", "Samsung"],
        "shape": [-1],
        "datatype": "STRING",
        "required": True,
    },
}
