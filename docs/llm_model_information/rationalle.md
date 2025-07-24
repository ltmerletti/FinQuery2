## Qwen/Qwen3-Embedding-0.6B
- Qwen3 Embedding 0.6B ranks 4th on the HuggingFace MTEB leaderboard, has exceptional performance, is fast, and small. It is a perfect model for this use case, and nets performance of above 90% on retrievals, following testing.

## arthurcollet/Qwen3-Reranker-0.6B-mlx-6bit
- A 6 bit quantization of the Qwen3 0.6B reranker for Apple Silicon. Retains a large amount of the original performance from Qwen3 Reranker, which ranks as the 4th best reranker on the HuggingFace MTEB leaderboard, and is very fast on Apple Silicon.

## qwen3-30b-a3b-mixed-3
- A 3 and 4 bit mixed quantization of the MoE Qwen 30B A3B. Data shows that highly quantized large models outperform small models often, and this one is incredibly fast to run (> 60 tok/s) while having a fairly high amount of intelligence and a high parameter count.

## qwen3-1.7b-mlx
- A very small, dense LLM. Consistently coherent and very fast (>100 tok/s), making it ideal for low-stakes work.

## gemma-3-text-27b-it
- The current best model for needle-in-a-haystack retrieval. Not perfect, but it is able to run on 16GB of VRAM, and offers the best performance as per Google's reported results for the task needed. It may be swapped as it is relatively slow (~25 tok/s).

## unsloth/granite-vision-3.2-2b-unsloth-bnb-4bit
- A quantized version of the granite vision dense model. It is recommended by Docling, and offers strong enough performance. Other contenders were Google's google/gemma-3n-e4b, but this has better instruction following. It is very fast, coherent, and will do the job for image explanation.