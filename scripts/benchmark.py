"""
scripts/benchmark.py — Performance benchmark script for Multi-Stream Video Intelligence.

Measures:
1. Frame sampling speed (OpenCV VideoCapture)
2. Detection latency (YOLO)
3. Tracking latency (ByteTrack)
4. CLIP embedding generation latency
5. FAISS vector search latency
6. Memory / GPU consumption
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.indexing.faiss_store import FaissStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def benchmark_faiss(n_vectors: int = 1000, dim: int = 512, top_k: int = 20):
    """Benchmark FAISS index build and search latency."""
    logger.info("--- FAISS Benchmark ---")
    store = FaissStore(dim=dim)
    
    # Generate dummy normalized embeddings
    embeddings = np.random.randn(n_vectors, dim).astype("float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    # Measure insertion
    t0 = time.perf_counter()
    store.add(embeddings)
    insert_duration = time.perf_counter() - t0
    logger.info(f"FAISS Insert {n_vectors} vectors: {insert_duration*1000:.2f} ms ({n_vectors/insert_duration:.1f} vec/sec)")
    
    # Measure search
    query_vec = np.random.randn(dim).astype("float32")
    query_vec = query_vec / np.linalg.norm(query_vec)
    
    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        results = store.search(query_vec, top_k=top_k)
        latencies.append((time.perf_counter() - t0) * 1000)
        
    avg_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)
    logger.info(f"FAISS Search (top-{top_k}) over {n_vectors} vectors: Mean={avg_lat:.3f} ms, P95={p95_lat:.3f} ms")


def benchmark_models():
    """Benchmark detector and CLIP encoder if available."""
    logger.info("--- Model Inference Benchmark ---")
    settings = get_settings()
    
    # Check GPU
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        logger.info(f"PyTorch Version: {torch.__version__}, CUDA Available: {cuda_avail}")
        if cuda_avail:
            logger.info(f"Device Name: {torch.cuda.get_device_name(0)}")
            logger.info(f"Allocated VRAM: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    except ImportError:
        logger.warning("Torch not installed yet.")
        return

    # Benchmark CLIP
    try:
        from src.embeddings.clip_encoder import ClipEncoder
        from PIL import Image
        
        logger.info(f"Loading CLIP ({settings.clip_model})...")
        t0 = time.perf_counter()
        encoder = ClipEncoder()
        load_time = time.perf_counter() - t0
        logger.info(f"CLIP loaded in {load_time:.2f} s on device: {encoder.device}")
        
        # Text embedding
        t0 = time.perf_counter()
        text_emb = encoder.encode_text("red car moving fast")
        text_lat = (time.perf_counter() - t0) * 1000
        logger.info(f"CLIP Text Embedding Latency: {text_lat:.2f} ms (dim={len(text_emb)})")
        
        # Image embedding (synthetic image)
        dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        t0 = time.perf_counter()
        img_emb = encoder.encode_image(dummy_img)
        img_lat = (time.perf_counter() - t0) * 1000
        logger.info(f"CLIP Image Embedding Latency: {img_lat:.2f} ms")
        
    except Exception as e:
        logger.warning(f"CLIP benchmark skipped: {e}")

    # Benchmark YOLO
    try:
        from src.detection.detector import ObjectDetector
        logger.info(f"Loading YOLO detector ({settings.yolo_model})...")
        t0 = time.perf_counter()
        detector = ObjectDetector()
        logger.info(f"YOLO loaded in {time.perf_counter() - t0:.2f} s")
        
        dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        latencies = []
        for _ in range(10):
            t0 = time.perf_counter()
            _ = detector.detect(dummy_frame)
            latencies.append((time.perf_counter() - t0) * 1000)
            
        logger.info(f"YOLO Detection Latency (640x640): Mean={np.mean(latencies):.1f} ms (~{1000/np.mean(latencies):.1f} FPS)")
    except Exception as e:
        logger.warning(f"YOLO benchmark skipped: {e}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Stream Video Intelligence Benchmark")
    parser.add_argument("--vectors", type=int, default=1000, help="Number of vectors for FAISS benchmark")
    args = parser.parse_args()
    
    logger.info("==================================================")
    logger.info("Multi-Stream Video Intelligence Benchmark")
    logger.info("==================================================")
    benchmark_faiss(n_vectors=args.vectors)
    benchmark_models()
    logger.info("==================================================")
    logger.info("Benchmark complete.")


if __name__ == "__main__":
    main()
