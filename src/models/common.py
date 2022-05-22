NOISE_TYPES = [
    'homo',
    'per-ens-homo',
    'hetero',
]

def raise_if_not_in_list(val, valid_options, varname):
    if val not in valid_options:
       msg = f'`{varname}` should be one of `{valid_options}` but was `{val}` instead.'
       raise RuntimeError(msg)
