"""Hybrid retrieval: dense vector search fused with a local BM25 lexical scorer.

Why this exists
---------------
Embedding-only retrieval is weak at "reference lookups" — questions like
"what is the formula in (15)", "explain Table 3", or "what does Section 4.2 say".
The reference token (``(15)``) carries almost no semantic signal, so the chunk
that actually contains it ranks low and falls outside ``similarity_top_k``.

A tiny in-memory BM25 scorer fixes this: a rare literal token like ``(15)`` has
high IDF, so the chunk containing it ranks first lexically. We fuse the vector
ranking and the lexical ranking with Reciprocal Rank Fusion (RRF) so both
signals contribute. This is fully local — no extra services or API keys — which
matters because the Cohere rerank key is optional/often unset.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from llama_index.core import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle

# Default answer prompt. The label-mapping paragraph is the key part: papers
# print equations as "(15)", so a synthesizer LLM (especially a strict one like
# Gemini) otherwise reports "no information about formula 15" even when the
# labelled expression is right there in the context.
DEFAULT_QA_TEMPLATE = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Answer the query using only the context above, not prior knowledge.\n\n"
    "IMPORTANT — reference labels: In documents, equations, formulas, tables, and "
    "figures are labelled with a parenthesised number such as (15). A question "
    "about 'formula 15', 'equation 15', 'eq. 15', or '(15)' refers to the "
    "expression or item labelled (15) in the context. Find that label and report "
    "the corresponding expression. Only say the answer is missing if no such "
    "label or content appears in the context.\n\n"
    "MATH DELIMITERS REQUIREMENT:\n"
    "For any mathematical formulas, equations, or symbols, format them in LaTeX STRICTLY:\n"
    "- Use $$...$$ on its OWN LINE for ALL block equations, matrices, multi-component "
    "formulas, or any formula with subscripts/superscripts spanning more than one symbol.\n"
    "- Use $...$ ONLY for single, short inline symbols like 'where $n$ is the count'.\n"
    "- NEVER use $...$ for multi-line expressions or matrices — use $$...$$ instead.\n"
    "- NEVER use plain brackets [ ] or parentheses ( ) for equations.\n"
    "- NEVER split a single formula across multiple $...$ inline spans.\n\n"
    "Query: {query_str}\n"
    "Answer: "
)

# Words + parenthesised numbers like "(15)" kept as single high-signal tokens.
_TOKEN_RE = re.compile(r"\(\d+\)|[a-z0-9]+")
# Reciprocal-rank-fusion constant. Smaller => a strong hit in ONE list matters
# more; 20 keeps a #1 lexical hit competitive with a top-5 vector hit.
_RRF_K = 20

# Words that, when followed by a number, name a labelled element in the paper.
_REF_WORDS = (
    r"formula|equations?|eqn?|tables?|figures?|fig|sections?|sec|theorems?|"
    r"lemmas?|corollar(?:y|ies)|algorithms?|alg|definitions?|def|"
    r"propert(?:y|ies)|chapters?|steps?|lines?|examples?"
)
_PAREN_NUM_RE = re.compile(r"\((\d+)\)")
_REF_NUM_RE = re.compile(rf"(?:{_REF_WORDS})\.?\s*\(?(\d+)\)?", re.IGNORECASE)


def _reference_numbers(query: str) -> set[str]:
    """Numbers the user is explicitly pointing at, e.g. 'formula 15' or '(27)'."""
    nums = set(_PAREN_NUM_RE.findall(query))
    nums.update(_REF_NUM_RE.findall(query))
    return nums


def _tokenize(text: str, *, is_query: bool = False) -> list[str]:
    # Documents print equation labels as "(15)". Users type "formula 15",
    # "eq 15", or "(15)". To make them match we expand asymmetrically:
    #   - in DOCS,  "(15)" also emits the bare number "15";
    #   - in QUERIES, a bare number "15" also emits the parenthesised "(15)".
    # Asymmetry matters: expanding bare doc numbers (page numbers, years) into
    # "(N)" would let a page-15 header match an "equation 15" query. This way
    # only the real "(15)" label carries the rare high-IDF token.
    out = []
    for tok in _TOKEN_RE.findall(text.lower()):
        out.append(tok)
        if tok.startswith("(") and tok.endswith(")"):
            out.append(tok[1:-1])
        elif is_query and tok.isdigit():
            out.append(f"({tok})")
    return out


class _HybridRetriever(BaseRetriever):
    """Fuse an index's dense retriever with a BM25 scorer over the same nodes."""

    def __init__(self, index, top_k: int):
        self._top_k = top_k
        # Pull a deeper candidate pool from each side than we ultimately return,
        # so a chunk that is (say) rank 11 by vector but rank 1 by BM25 survives.
        pool = max(top_k * 3, 20)
        self._vector = index.as_retriever(similarity_top_k=pool)
        self._pool = pool

        self._nodes = list(index.docstore.docs.values())
        self._contents = [n.get_content() for n in self._nodes]
        self._tokens = [_tokenize(c) for c in self._contents]
        n_docs = len(self._nodes) or 1
        self._avgdl = sum(len(t) for t in self._tokens) / n_docs
        df: Counter = Counter()
        for toks in self._tokens:
            df.update(set(toks))
        self._idf = {
            term: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        super().__init__()

    def _bm25_scores(self, query: str, k1: float = 1.5, b: float = 0.75) -> list[float]:
        q_terms = _tokenize(query, is_query=True)
        scores = []
        for toks in self._tokens:
            tf = Counter(toks)
            dl = len(toks) or 1
            s = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                idf = self._idf.get(term, 0.0)
                s += idf * tf[term] * (k1 + 1) / (tf[term] + k1 * (1 - b + b * dl / self._avgdl))
            scores.append(s)
        return scores

    def _lexical(self, query: str) -> list[NodeWithScore]:
        scores = self._bm25_scores(query)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in ranked[: self._pool]:
            if scores[i] <= 0:
                break
            out.append(NodeWithScore(node=self._nodes[i], score=scores[i]))
        return out

    def _referenced_nodes(self, query: str) -> list:
        """Nodes whose text literally contains a label the query names, e.g. '(15)'.

        This is the deterministic guarantee: an 'equation 15' question always
        pulls in the chunk printing '(15)', which pure similarity ranking can
        drop when the query and chunk share little other wording.
        """
        nums = _reference_numbers(query)
        if not nums:
            return []
        labels = [f"({n})" for n in nums]
        return [
            node
            for node, content in zip(self._nodes, self._contents)
            if any(lbl in content for lbl in labels)
        ]

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query = query_bundle.query_str
        vec = self._vector.retrieve(query_bundle)
        lex = self._lexical(query)

        # Reciprocal Rank Fusion, keeping a handle on each node for the result.
        fused: dict[str, float] = {}
        keep: dict[str, NodeWithScore] = {}
        for ranked in (vec, lex):
            for rank, nws in enumerate(ranked):
                nid = nws.node.node_id
                fused[nid] = fused.get(nid, 0.0) + 1.0 / (_RRF_K + rank)
                keep.setdefault(nid, nws)

        order = sorted(fused, key=lambda nid: fused[nid], reverse=True)

        # Force explicitly-referenced chunks ('(15)') to the front, then fill the
        # rest of the top-k with the fused ranking.
        forced = self._referenced_nodes(query)
        top_score = fused[order[0]] if order else 1.0
        results: list[NodeWithScore] = []
        seen: set[str] = set()
        for i, node in enumerate(forced[: self._top_k]):
            results.append(NodeWithScore(node=node, score=top_score + len(forced) - i))
            seen.add(node.node_id)
        for nid in order:
            if len(results) >= self._top_k:
                break
            if nid in seen:
                continue
            results.append(NodeWithScore(node=keep[nid].node, score=fused[nid]))
            seen.add(nid)
        return results


# Cache built retrievers so the BM25 side-index (tokenise every chunk + IDF
# table) is computed ONCE per document set, not on every query. The BM25 stats
# depend only on the chunks, never on the query, so they are safe to reuse; the
# per-query relevance scoring still runs fresh on each retrieve() call. Keyed by
# the set of node ids (stable across index reloads from disk) + top_k.
_retriever_cache: dict[str, _HybridRetriever] = {}


def _index_signature(index, top_k: int) -> str:
    ids = "".join(sorted(index.docstore.docs.keys()))
    return f"{hashlib.sha1(ids.encode()).hexdigest()}_{top_k}"


def make_retriever(index, *, similarity_top_k) -> _HybridRetriever:
    """Return a hybrid retriever, reusing a cached one when the chunks match."""
    key = _index_signature(index, similarity_top_k)
    retriever = _retriever_cache.get(key)
    if retriever is None:
        retriever = _HybridRetriever(index, top_k=similarity_top_k)
        _retriever_cache[key] = retriever
    return retriever


def format_context(nodes: list[NodeWithScore]) -> str:
    """Render retrieved chunks as plain text for an agent to read directly."""
    if not nodes:
        return "No matching passages were found in the document."
    parts = []
    for i, nws in enumerate(nodes, 1):
        meta = nws.node.metadata or {}
        page = meta.get("page") or meta.get("page_label") or "?"
        parts.append(f"[Excerpt {i} | page {page}]\n{nws.node.get_content()}")
    return "\n\n".join(parts)


def make_query_engine(index, *, similarity_top_k, text_qa_template=None, llm=None):
    """RetrieverQueryEngine backed by the hybrid vector+BM25 retriever."""
    retriever = make_retriever(index, similarity_top_k=similarity_top_k)
    kwargs = {"text_qa_template": text_qa_template or DEFAULT_QA_TEMPLATE}
    if llm is not None:
        kwargs["llm"] = llm
    return RetrieverQueryEngine.from_args(retriever, **kwargs)
