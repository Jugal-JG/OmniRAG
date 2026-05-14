from engines.basic_rag import run as basic_rag
from engines.multi_document import run as multi_document
from engines.multimodal import run as multimodal
from engines.react_agent import run as react
from engines.router_engine import run as router_engine
from engines.subquestion import run as subquestion

ENGINES = {
    "basic_rag": basic_rag,
    "multi_document": multi_document,
    "multimodal": multimodal,
    "react": react,
    "router_engine": router_engine,
    "subquestion": subquestion,
}


def get_engine(label: str):
    return ENGINES.get(label, basic_rag)
