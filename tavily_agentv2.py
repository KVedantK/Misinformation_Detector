import os
import json
from tavily import TavilyClient
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

HF_TOKEN    = ""
TAVILY_KEY  = ""

os.environ["HF_TOKEN"]                 = HF_TOKEN
os.environ["TAVILY_API_KEY"]           = TAVILY_KEY
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ["HF_TOKEN"]

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def make_llm() -> ChatHuggingFace:
    endpoint = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.3-70B-Instruct",
        provider="auto",
        huggingfacehub_api_token=os.environ["HF_TOKEN"],
        max_new_tokens=800,
        temperature=0.01,
        timeout=120,
    )
    return ChatHuggingFace(llm=endpoint, verbose=False)

llm = make_llm()

QUERY_PROMPT = SystemMessage(content="""You are a search query generator.
Given a news article, generate ONE concise search query for a search engine
that would find evidence to verify or debunk it. Do not be too general we need to specifically get infomration arround the article not general infromation.
Output only the query — no explanation, no punctuation at the end.""")

query_chain = llm | StrOutputParser()

def get_evidence(title: str) -> dict:
    print(f"  [Chain 1] Generating search query...")

    query_raw = query_chain.invoke([
        QUERY_PROMPT,
        HumanMessage(content=f"News article: {title}")
    ])
    query = query_raw.strip().strip('"').strip("'")
    print(f"  [Chain 1] Query: {query}")

    response = tavily_client.search(
        query=query,
        max_results=10,
        search_depth="advanced",
        include_answer=True
    )

    snippets = []
    sources  = []
    for r in response.get("results", []):
        snippets.append(
            f"Source: {r['url']}\n"
            f"Title:  {r['title']}\n"
            f"Text:   {r['content'][:800]}"
        )
        sources.append(r["url"])

    tavily_answer = response.get("answer", "")

    evidence_text = "\n\n---\n\n".join(snippets)

    print(f"  [Chain 1] Found {len(snippets)} sources")

    return {
        "query":          query,
        "evidence_text":  evidence_text,
        "tavily_answer":  tavily_answer,
        "sources":        sources,
    }



VERDICT_SYSTEM = SystemMessage(content="""You are a fact-checking assistant.
You will be given a news article and web evidence retrieved about it.
Your job is to decide if the evidence supports or contradicts the article.
Respond ONLY with valid JSON in exactly this format:
{
  "verdict": "SUPPORTED" or "CONTRADICTED",
  "confidence": <number 0.0 to 1.0>,
  "explanation": "<one sentence explaining your decision>",
  "supporting_sources": ["<url>"],
  "contradicting_sources": ["<url>"]
}
Output only the JSON. No preamble. No explanation outside the JSON.""")

verdict_chain = llm | StrOutputParser()

def get_verdict(title: str, evidence: dict) -> dict:
    """
    Chain 2:
      title + evidence → LLM reads and judges → parsed JSON verdict
    """
    print(f"  [Chain 2] Generating verdict...")

    user_message = HumanMessage(content=(
        f"News article to verify:\n{title}\n\n"
        f"Tavily summary: {evidence['tavily_answer']}\n\n"
        f"Web evidence:\n{evidence['evidence_text']}\n\n"
        f"Based on this evidence, is the article supported or contradicted?"
    ))

    raw = verdict_chain.invoke([VERDICT_SYSTEM, user_message])

    verdict     = "INSUFFICIENT_EVIDENCE"
    confidence  = 0.5
    explanation = raw[:200]

    try:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start >= 0 and end > start:
            parsed      = json.loads(raw[start:end])
            verdict     = parsed.get("verdict",     verdict)
            confidence  = float(parsed.get("confidence", confidence))
            explanation = parsed.get("explanation", explanation)
    except Exception as e:
        print(f"  [Chain 2] JSON parse warning: {e}")
        print(f"  [Chain 2] Raw: {raw[:300]}")

    print(f"  [Chain 2] Verdict: {verdict} ({confidence:.2f})")

    return {
        "verdict":     verdict,
        "confidence":  confidence,
        "explanation": explanation,
    }


def verify_article(title: str) -> dict:
    print(f"\n{'='*60}")
    print(f"Title: {title[:80]}")
    print(f"{'='*60}")


    evidence = get_evidence(title)

    verdict  = get_verdict(title, evidence)

    return {
        "title":       title,
        "query":       evidence["query"],
        "sources":     evidence["sources"],
        **verdict
    }

