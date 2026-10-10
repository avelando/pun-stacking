def input_prompt_for_model(model_name: str) -> str:
    if model_name == "intfloat/multilingual-e5-large":
        return "query: "

    if model_name == "google/embeddinggemma-300m":
        return "task: classification | query: "

    return ""
