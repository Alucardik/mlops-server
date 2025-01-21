import json

from numpy import array, float32
from onnxruntime import NodeArg


cat_cols = {
    "neighbourhood",
    "room_type",
    "property_type",
}

num_cols = {
    "bedrooms",
    "accommodates",
    "bathrooms",
    "minimum_nights",
    "host_listings_count",
    "beds",
}

default_true = frozenset(("availability_30", "availability_60", "availability_90", "availability_365", "host_is_superhost"))
default_85 = frozenset(("number_of_reviews_ltm", "review_scores_rating"))
default_custom = {
    "latitude": 41.40889,
    "longitude": 2.18555,
}


def wrap_num_param(param):
    return [[array(param, dtype=float32).tolist()]]


def wrap_cat_param(param):
    return [[param]]


def assemble_model_input(body: dict, inputs: list[NodeArg]) -> dict:
    selected_amenities = set(body.get("amenities", []))

    model_input = {}
    for inp in inputs:
        input_name = inp.name
        if input_name in default_custom:
            model_input[input_name] = wrap_num_param(default_custom[input_name])
            continue
        if input_name in default_true:
            model_input[input_name] = wrap_num_param(1.)
            continue
        if input_name in default_85:
            model_input[input_name] = wrap_num_param(85.)
            continue
        if input_name.startswith("has_"):
            model_input[input_name] = wrap_num_param(1 if input_name[4:] in selected_amenities else 0)
            continue
        if input_name in num_cols:
            model_input[input_name] = wrap_num_param(body.get(input_name))
            continue
        if input_name in cat_cols:
            model_input[input_name] = wrap_cat_param(body.get(input_name))
            continue

    return model_input