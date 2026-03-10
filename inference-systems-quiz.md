# LLM Inference Systems — Quiz

**Instructions:** Fill in your answers below each question. When done, send me a message and I'll grade everything.

---

## Section 1: Two Phases (10 pts)

**Q1 (2 pts):** What are the two phases of LLM inference, and what is each phase bound by?

**Your answer:**
Prefill and decoding. Prefill is bounded by computation, decoding is bounded by memory

---

**Q2 (3 pts):** A Llama 70B model is running on a single A100 (312 TFLOPS FP16, 2 TB/s bandwidth). What is the approximate roofline crossover batch size where decode goes from memory-bound to compute-bound? Show your reasoning.

**Your answer:**
In decoding, 1 token requires 70B flops (ignore attention score) = 70,000,000,000 op, to fill 312 TFLOPS, we need 312 * 1024 * 1024 * 1024 * 1024 =  4900 tokens 

Decoding 1 token requires ... (not sure)

---

**Q3 (2 pts):** What do TTFT and TTBT stand for, and which phase does each correspond to?

**Your answer:**
TTFT = Time To First Token -> corresponding to prefill phase
TTBT = Time Between Token -> corresponding to decoding phase

---

**Q4 (3 pts):** Why does batching help the decode phase so much but barely affects prefill?

**Your answer:**
Because prefill is compute bound, (not sure what to follow)

---

## Section 2: Memory Math (15 pts)

**Q5 (3 pts):** How much memory do Llama 70B weights consume in FP16? In INT8?

**Your answer:**
70B * 2 = 140B in FP16
70B in INT8

---

**Q6 (4 pts):** Calculate the KV cache memory per token for Llama 70B in FP16. The model has 80 layers, 8 KV heads, and head dimension 128. Show your formula.

**Your answer:**
per token KV cache memory = 2 * 80 * 8 * 128 * 2 = 327680 bytes (question: why do we not need to check embedding size?)

---

**Q7 (4 pts):** A Llama 70B (FP16) serving system has 4× A100 80GB (320 GB total). After loading weights and accounting for ~10% overhead, how many concurrent 4K-context requests can it serve? Show your math.

**Your answer:**
320GB - 70GB = 250GB
320GB * 10% = 32

250-32 = 218 GB

4k * 327680  = 1.34GB

218 /1.34 ~= 162 requests

---

**Q8 (4 pts):** Without FlashAttention, how much memory does the attention score matrix consume per layer for Llama 70B at sequence length 4096 (FP16)? Why is this problematic, and how does FlashAttention fix it?

**Your answer:**
4096 * 4096 * 2 = 33MB

It's problematic because it's O(S^2)

Flash Attention addresses this by tiling the matrix and utilize shared memory instead of global memory to perform the matrix multiplication, then use online software trick to do softmax without materializing the matrix in memory

---

## Section 3: Batching (10 pts)

**Q9 (3 pts):** Explain continuous batching (Orca) and why it's superior to static batching.

**Your answer:**
Static batching works simply by batching all admined tasks and run until all finishes. 
Continuous batching works by scheduling on iteration steps. It's basically asking every decoding task "are you finished" at each iteration step. And if so, evict the KV cache and put in another task. 

Continuous batching is superior because it eliminates GPU idling situation where a 500 token response has to wait for a 5000 token response to fully decode before returning to user.

---

**Q10 (4 pts):** What problem does chunked prefill (Sarathi) solve? Describe the mechanism.

**Your answer:**
Chunked prefill works by segmenting the prefill to 512 tokens per segment and interleave them with the decoding task. This solves the problem of prefill influencing TTBT of existing decoding jobs.

---

**Q11 (3 pts):** A serving system uses continuous batching with a token budget of 2048 per step. There are 20 active decode requests and a new request arrives with a 6000-token prompt. How is this handled?

**Your answer:**
We segment the 6000-token prompt into roughly 12 prefill jobs (each taking 500 tokens). Then we interleave it with the existing decode requests as soon as one active decode requests finishes. (Q: they share weight but what happens if KV cache spills?)

---

## Section 4: PagedAttention (12 pts)

**Q12 (4 pts):** Explain the problem PagedAttention solves and how it works. Use a memory analogy.

**Your answer:**
PagedAttention essentially mimics how operating systems solves the problem of CPU memory using virtual memory and block table. PagedAttention divides the physical memory of a GPU into fixed size block and creates a layer of virtual memory block with mapping to the physical block, along with reference counting. This reduces external fragmentation compared to prior solutions where a small request might allocate a large, fixed size block of GPU memory.

---

**Q13 (4 pts):** A request has generated 145 tokens with a block size of 16. How many blocks are allocated? What is the internal fragmentation? How does this compare to pre-allocation for a max_seq_len of 2048?

10 blocks? close to 0.94 block of internal fragmentation -> around 15 tokens of internal fragmentation. It's way better than pre-allocation of 2048 because it would have wasted 2048-145=1903 tokens
**Your answer:**

---

**Q14 (4 pts):** How does PagedAttention enable efficient beam search? What is copy-on-write in this context?
During beam search, we keep top k best sequence and they those sequence often share the same prefix, and because the shared prefix, they share common KV cache block. Copy-on-write happens when the next predicted token diverges and we would copy the KV cache block that contains the token that diverges. We might evict those block when we determine that the sequence has dropped outside the the top k

**Your answer:**

---

## Section 5: Parallelism (12 pts)

**Q15 (3 pts):** What is tensor parallelism (TP)? Why does it require fast interconnect?
TP is when we split the model weights across multiple GPUs. It requires fast interconnect like NCCL because we need to perform 2 all-reduce operations each layer, one before the layer norm and one before the ??

**Your answer:**

---

**Q16 (3 pts):** What is the bubble ratio in pipeline parallelism? With P=4 stages and M=12 micro-batches, what is it?

**Your answer:**
Can't do the math

---

**Q17 (3 pts):** How does Ring Attention work for long sequences? Which production system uses it?

**Your answer:**
Ring attention works when the context is long, it works by having each GPU holds only partial attention and having a ring-based computation (GPU1 -> GPU2 -> GPU3...) without ever materializing the entire attention score. Gemini uses it to achieve its 1M context size.

---

**Q18 (3 pts):** You need to serve Llama 405B for inference. Weights are 810 GB in FP16. You have H100 80GB GPUs. What's the minimum GPU count, and what parallelism strategy would you use?

**Your answer:**
To hold weight of 810GB we need at least 9 GPU, but that leaves very small room for holding KV cache. I'd say 10 GPUs. We have to use TP

---

## Section 6: Key Optimizations (15 pts)

**Q19 (3 pts):** Explain speculative decoding. What determines the speedup?

**Your answer:**
Speculative decoding 

---

**Q20 (3 pts):** What is GQA and how does it reduce memory usage compared to MHA? What's the KV cache reduction factor for Llama 70B (64 Q heads, 8 KV heads)?

**Your answer:**
Group Query ATtention. INstead of multi-head attention, we reduce the number of heads by assuming that one head can reprsent a multi-dimensional aspect of the context. 
The reduction factor is 8.

---

**Q21 (3 pts):** What are CUDA Graphs and why do they help inference? What's the main constraint?

**Your answer:**
Cuda graph is the computation graph. We can prune the unusued path to save (sorry Idk)

---

**Q22 (3 pts):** Explain prefix caching (RadixAttention). When does it provide the most benefit?

**Your answer:**
RadixAttention is SGLang's method compared to PagedAttention, what they do is they build a radix / prefix tree, and using that tree to manage KV cache block by checking if sequence share the same prefix.
It provides the most benefit for decoding in a chatbot because chatbot often have the same system prompt and that prompt's KV cache can be shared.

---

**Q23 (3 pts):** Compare FP16, FP8, INT8, and INT4 quantization in terms of memory savings and quality impact.

**Your answer:**
2bytes, 1byte, 1byte (int representation), half a byte

INT8 is better than FP8, IN4 has the wrst quality 

---

## Section 7: Long Context & Attention (8 pts)

**Q24 (4 pts):** What are "attention sinks" and why does StreamingLLM keep them?

**Your answer:**
No idea

---

**Q25 (4 pts):** Name three approaches for extending a model's context length beyond its training window. Briefly explain each.

**Your answer:**
No idea

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
10TB/s ?

**Q29:** H100 80GB FP16 TFLOPS?
**Your answer:**
500TFLOPs?

**Q30:** Flash Attention memory complexity?
**Your answer:**
O(s), s is the sequence length

**Q31:** Typical speedup from continuous batching vs static batching?
**Your answer:**
2-3x

**Q32:** Typical speedup from speculative decoding?
**Your answer:**
2-3x

**Q33:** Llama 70B weights in FP16 — minimum number of A100 80GB GPUs?
**Your answer:**
2

**Q34:** What autoscaling signal should you use instead of GPU utilization?
**Your answer:**
TTFT and TTBT

**Q35:** PagedAttention typical memory utilization improvement?
**Your answer:**
3-4x

---

**Total: 100 points**

*Good luck! Fill in your answers and let me know when you're ready for grading.*
