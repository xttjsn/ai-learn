# LLM Inference Systems — Quiz

**Instructions:** Fill in your answers below each question. When done, send me a message and I'll grade everything.

---

## Section 1: Two Phases (10 pts)

**Q1 (2 pts):** What are the two phases of LLM inference, and what is each phase bound by?

**Your answer:**

---

**Q2 (3 pts):** A Llama 70B model is running on a single A100 (312 TFLOPS FP16, 2 TB/s bandwidth). What is the approximate roofline crossover batch size where decode goes from memory-bound to compute-bound? Show your reasoning.

**Your answer:**

---

**Q3 (2 pts):** What do TTFT and TTBT stand for, and which phase does each correspond to?

**Your answer:**

---

**Q4 (3 pts):** Why does batching help the decode phase so much but barely affects prefill?

**Your answer:**

---

## Section 2: Memory Math (15 pts)

**Q5 (3 pts):** How much memory do Llama 70B weights consume in FP16? In INT8?

**Your answer:**

---

**Q6 (4 pts):** Calculate the KV cache memory per token for Llama 70B in FP16. The model has 80 layers, 8 KV heads, and head dimension 128. Show your formula.

**Your answer:**

---

**Q7 (4 pts):** A Llama 70B (FP16) serving system has 4× A100 80GB (320 GB total). After loading weights and accounting for ~10% overhead, how many concurrent 4K-context requests can it serve? Show your math.

**Your answer:**

---

**Q8 (4 pts):** Without FlashAttention, how much memory does the attention score matrix consume per layer for Llama 70B at sequence length 4096 (FP16)? Why is this problematic, and how does FlashAttention fix it?

**Your answer:**

---

## Section 3: Batching (10 pts)

**Q9 (3 pts):** Explain continuous batching (Orca) and why it's superior to static batching.

**Your answer:**

---

**Q10 (4 pts):** What problem does chunked prefill (Sarathi) solve? Describe the mechanism.

**Your answer:**

---

**Q11 (3 pts):** A serving system uses continuous batching with a token budget of 2048 per step. There are 20 active decode requests and a new request arrives with a 6000-token prompt. How is this handled?

**Your answer:**

---

## Section 4: PagedAttention (12 pts)

**Q12 (4 pts):** Explain the problem PagedAttention solves and how it works. Use a memory analogy.

**Your answer:**

---

**Q13 (4 pts):** A request has generated 145 tokens with a block size of 16. How many blocks are allocated? What is the internal fragmentation? How does this compare to pre-allocation for a max_seq_len of 2048?

**Your answer:**

---

**Q14 (4 pts):** How does PagedAttention enable efficient beam search? What is copy-on-write in this context?

**Your answer:**

---

## Section 5: Parallelism (12 pts)

**Q15 (3 pts):** What is tensor parallelism (TP)? Why does it require fast interconnect?

**Your answer:**

---

**Q16 (3 pts):** What is the bubble ratio in pipeline parallelism? With P=4 stages and M=12 micro-batches, what is it?

**Your answer:**

---

**Q17 (3 pts):** How does Ring Attention work for long sequences? Which production system uses it?

**Your answer:**

---

**Q18 (3 pts):** You need to serve Llama 405B for inference. Weights are 810 GB in FP16. You have H100 80GB GPUs. What's the minimum GPU count, and what parallelism strategy would you use?

**Your answer:**

---

## Section 6: Key Optimizations (15 pts)

**Q19 (3 pts):** Explain speculative decoding. What determines the speedup?

**Your answer:**

---

**Q20 (3 pts):** What is GQA and how does it reduce memory usage compared to MHA? What's the KV cache reduction factor for Llama 70B (64 Q heads, 8 KV heads)?

**Your answer:**

---

**Q21 (3 pts):** What are CUDA Graphs and why do they help inference? What's the main constraint?

**Your answer:**

---

**Q22 (3 pts):** Explain prefix caching (RadixAttention). When does it provide the most benefit?

**Your answer:**

---

**Q23 (3 pts):** Compare FP16, FP8, INT8, and INT4 quantization in terms of memory savings and quality impact.

**Your answer:**

---

## Section 7: Long Context & Attention (8 pts)

**Q24 (4 pts):** What are "attention sinks" and why does StreamingLLM keep them?

**Your answer:**

---

**Q25 (4 pts):** Name three approaches for extending a model's context length beyond its training window. Briefly explain each.

**Your answer:**

---

## Section 8: System Design (10 pts)

**Q26 (5 pts):** You're designing a production LLM serving system for a chatbot with 10K concurrent users, each with multi-turn conversations averaging 2K tokens of context. The model is Llama 70B. Describe your architecture: hardware, parallelism, key optimizations, and what metrics you'd monitor.

**Your answer:**

---

**Q27 (5 pts):** Your serving system's P99 TTBT has spiked from 25ms to 120ms. Walk through your debugging process. What are the likely causes and how would you investigate each?

**Your answer:**

---

## Section 9: Quick Fire — Numbers (8 pts, 1 pt each)

**Q28:** A100 80GB HBM bandwidth?
**Your answer:**

**Q29:** H100 80GB FP16 TFLOPS?
**Your answer:**

**Q30:** Flash Attention memory complexity?
**Your answer:**

**Q31:** Typical speedup from continuous batching vs static batching?
**Your answer:**

**Q32:** Typical speedup from speculative decoding?
**Your answer:**

**Q33:** Llama 70B weights in FP16 — minimum number of A100 80GB GPUs?
**Your answer:**

**Q34:** What autoscaling signal should you use instead of GPU utilization?
**Your answer:**

**Q35:** PagedAttention typical memory utilization improvement?
**Your answer:**

---

**Total: 100 points**

*Good luck! Fill in your answers and let me know when you're ready for grading.*
