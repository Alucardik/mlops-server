from numpy import array, float32


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


def wrap_num_param(param):
    return [[array(param, dtype=float32)]]


def wrap_cat_param(param):
    return [[param]]