from importlib import import_module


ENGINES = {
    "basic_rag": "engines.basic_rag",
    "multi_document": "engines.multi_document",
    "multimodal": "engines.multimodal",
    "react": "engines.react_agent",
    "router_engine": "engines.router_engine",
    "subquestion": "engines.subquestion",
}


def get_engine(label: str):
    module_name = ENGINES.get(label, ENGINES["basic_rag"])
    return import_module(module_name).run
