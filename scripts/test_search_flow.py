import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.query.parser import QueryParser
from src.query.retriever import RetrievalPipeline

def test_queries():
    parser = QueryParser()
    pipeline = RetrievalPipeline()

    queries = ['find cars', 'show me people', 'motorcycle', 'find truck', 'green car']
    for q in queries:
        sq = parser.parse(q)
        results, clar = pipeline.search(sq, top_k=5)
        print(f"\n=======================================================")
        print(f"Query: '{q}' | Parsed: class={sq.object_class}, attrs={sq.attributes}")
        print(f"Found: {len(results)} matches")
        for r in results:
            print(f"  -> [{r.camera_id}] at t={r.timestamp:.1f}s | conf={r.confidence:.2f} | score={r.rerank_score:.2f} | class={r.object_class} | color={r.dominant_color}")

if __name__ == "__main__":
    test_queries()
