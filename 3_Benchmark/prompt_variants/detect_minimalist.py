from benchmark.prompt_variations import get_variation_prompts

mods = ("image", "video", "audio")
variant = {m: get_variation_prompts(m)["minimalist"] for m in mods}

SYSTEM_HINT = variant["image"]["system"]
questions = [f"<{m}>\n{variant[m]['user']}" for m in mods]
