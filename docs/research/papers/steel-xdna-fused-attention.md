# STEEL: Sparsity-Aware Fused Attention for Energy-Efficient Long-Sequence Inference on AMD's XDNA NPU

arXiv: 2607.09385

## Abstract
The growing adoption of large language model-based agents within operating system workflows has increased the importance of energy-efficient inference on laptop-class systems-on-chip (SoCs). While cloud offloading remains common, it introduces reliability and privacy concerns that are particularly problematic for agentic workloads. Recent laptop SoCs, therefore, incorporate neural processing engines (NPUs) optimized for energy efficiency; however, effectively mapping attention mechanisms onto NPUs remains challenging due to architectural diversity and explicit data-movement programming models. In this work, we present STEEL, the first open-source implementation of FlashAttention targeting XDNA-like NPUs. STEEL introduces a dataflow formulation of prefill attention, enabling efficient exploitation of spatial parallelism and on-chip memory. Furthermore, STEEL addresses the load imbalance induced by the causal mask by leveraging a sparsity-aware pipeline placement onto the NPU array, reducing synchronization overhead and improving utilization. We evaluate STEEL on the AMD Ryzen AI 9 HX 370 SoC and compare its performance against optimized CPU and GPU implementations. Experimental results show that STEEL reduces energy consumption by an average of 9.17x and 1.75x relative to CPU and GPU baselines, respectively. On XDNA 1, STEEL achieves an average 9.6x latency reduction over the prior state of the art, and delivers a 22.8x speedup on average compared to a layer-by-layer attention implementation on XDNA 2.

## I Introduction

The increasing integration of artificial intelligence (AI) agents into core operating system functions is a key driver in the design of modern laptop systems-on-chip (SoCs) ([13]). These agents are typically implemented as large Transformer-based deep neural networks (DNNs) with several billion parameters ([20]). While such models enable powerful capabilities, their inference demands impose substantial computational and data-movement overhead, making them inherently energy-intensive. This energy cost has emerged as a fundamental bottleneck for embedded mobile platforms, where power and thermal budgets are tightly constrained ([20]).

As a result, most large language model (LLM) inference is currently offloaded to data-center graphics processing units (GPUs). Although effective from a performance standpoint, this centralized approach introduces challenges in agentic workflows, including increased latency, reduced reliability, and heightened privacy risks ([2]).

To unlock the full potential of AI agents at the edge, recent laptop SoCs integrate neural processing units (NPUs) designed specifically for energy-efficient inference ([19, 12, 7]). NPUs target the most computationally and energy expensive components of Transformer models, most notably the attention mechanism ([23]).
The prefill stage of Attention is a major contributor to inference latency and energy consumption at long-sequence length, a regime that is increasingly common in practical LLM deployments ([23]).
As a result, substantial effort has been made to optimize attention on both commercial ([4]) and academic hardware platforms ([10]). The range of attention optimizations is broad, from algorithmic improvements like FlashAttention ([4]) to hardware enhancements including specialized non-linear units ([17]).

NPUs achieve high energy efficiency through spatial dataflow architectures and explicit data-movement programming models, which expose fine-grained control over computation and memory transfers ([9]).
Typical NPU architectures, such as AMD’s XDNA™, are particularly well-suited for executing the prefill stage of language generation, as the prefill stage is primarily composed of matrix-to-matrix multiplication. Additionally, the prefill stage is a significant contributor to latency for large context requests ([1]), as is the case in agentic systems.
While extensive prior work has focused on optimizing attention for GPUs, comparatively few efforts target attention on NPUs ([12]). Moreover, the substantial diversity among NPUs architectures and programming models severely limits the portability of existing solutions.

In this work, we present STEEL, the first open-source implementation of FlashAttention on XDNA™ 2 NPUs.
STEEL proposes a dataflow formulation of the prefill attention distributed onto a three-stage pipeline of artificial intelligence engine (AIE) tiles. Through a careful distribution of work over the pipeline, optimized data-layout handling, and sparsity-aware pipeline placement, STEEL enhances the energy efficiency of the attention mechanism on edge platforms.
This work makes the following key contributions:

* •

We propose STEEL, a dataflow formulation of FlashAttention for XDNA™-like architectures. STEEL decomposes the FlashAttention algorithm to enable efficient distribution across the processing element (PE) array.
* •

We introduce a novel sparsity-aware pipeline placement technique that mitigates workload distribution imbalance caused by the causal masking, hence reducing synchronization overhead. This placement achieves a 38 % latency reduction compared to uniform placement.
* •

We perform an in-depth comparison of attention execution on the AMD Ryzen™ AI 9 HX 370 SoC across its NPU, central processing unit (CPU), and GPU. In detail, we benchmark STEEL on the XDNA™ 2 NPU and compare it against FlashAttention on 12 Zen5 CPUs cores and the RDNA 3.5 GPU.
* •

We demonstrate STEEL’s portability across multiple XDNA™-like architectures showing that it outperforms the state of the art (SotA) FlashAttention implementation from DATO ([6]).

Experimental evaluations of STEEL on AMD Ryzen™ AI 9 HX 370 demonstrate an average energy consumption reduction of 9.17 ×\times and 1.75 ×\times compared to the CPU and GPU, respectively.
On XDNA™ 1, STEEL outperforms the previous SotA implementation of flash-attention on XDNA™ 1 ([6]) by reducing the latency by 9.6 ×\times on average.
Additionally, compared to a layer-by-layer implementation of attention on XDNA™ 2, STEEL provides an average 22.8 ×\times speedup.
The STEEL algorithm is open-source at https://github.com/amd/iron.

## II Background

### II-A Fused Attention Algorithms

Transformer blocks form the computational backbone of essentially all LLMs and are used in a large fraction of modern large-scale DNNs ([8]). Within each transformer block, the *attention mechanism* ([22]) is responsible for much of the computational cost, memory footprint, and performance complexity.
As sequence lengths grow, attention often becomes the dominant contributor to inference latency and memory bottlenecks ([23]).

The attention mechanism operates on three input tensors: queries Q∈ℝSq×dQ\in\mathbb{R}^{S\_{q}\times d}  keys K∈ℝSk​v×dK\in\mathbb{R}^{S\_{kv}\times d}, and values V∈ℝSk​v×dV\in\mathbb{R}^{S\_{kv}\times d}, and produces an output tensor O∈ℝSq×dO\in\mathbb{R}^{S\_{q}\times d}:

|  | A=Q​KTd,O=s​o​f​t​m​a​x​(A)​V​where​A∈ℝSq×Sk​vA=\frac{QK^{T}}{\sqrt{d}},\;O=softmax(A)V\;\text{where}\;A\in\mathbb{R}^{S\_{q}\times S\_{kv}}{\color[rgb]{0,0,0}{}} |  | (1) |
| --- | --- | --- | --- |

Where SqS\_{q} and Sk​vS\_{kv} denote the query and key/value sequence lengths, respectively, and dd denotes the head dimension. The softmax function is applied independently across each row of A.

FlashAttention ([4]) was introduced to eliminate the storage and latency overheads incurred by materializing AA and s​o​f​t​m​a​x​(A)softmax(A) in memory. Its key insight is to restructure the attention computation to eliminate explicit materialization of these intermediates entirely. FlashAttention achieves this by leveraging online softmax techniques ([14]) and tiling the attention computation, allowing the attention score to be computed, normalized, and consumed incrementally.
To preserve the correct normalization statistics across tiles, FlashAttention maintains two per-row statistics vectors, ℓ\ell m∈ℝBqm\in\mathbb{R}^{B\_{q}}, where BqB\_{q} is the user-selected block size for QQ.

### II-B XDNA™  Software Stack

To program the XDNA™ 2 NPU, we use AMD’s open-source software stack (Fig. 1). IRON provides compute primitives targeting both individual AIE cores and the full NPU. Single-core primitives (kernels) are implemented in C++ using the AIE API, while full-NPU primitives (designs) are implemented in Python via MLIR-AIE bindings. Designs compose kernels and orchestrate data movement across AIE cores, Mem tiles, and dynamic random-access memory (DRAM). For instance, the IRON general matrix multiplication (GEMM) design distributes tiles across the array while invoking a C++ kernel for local computation.

![Figure 1: Overview of the software stack used to program the XDNA™ 2 NPU. The IRON library contains efficient ML operators written in Python and using C++ kernels. The Python bindings are lowered to LLVM IR by the MLIR-AIE compiler. The LLVM-AIE compiler generates binaries to run on the NPU; the host-to-NPU interactions are handled by the XRT runtime. The entire stack is composed of open-source tools.](https://arxiv.org/html/2607.09385/2607.09385v1/x1.png)

The Python bindings define the dataflow over the PE array and are lowered to the MLIR-AIE dialect, optimized through transformation passes, and translated to LLVM IR. This IR is compiled by LLVM-AIE into binaries for the AIE cores. Execution is managed by a C++ or Python runtime built on XRT or PyXRT, which either runs the operator on the NPU and returns output tensors or executes validation test benches.

### II-C XDNA™ 2 NPU Architecture

![Figure 2: Overview of the XDNA™ 2 NPU. XDNA™ 2 features three types of tiles interconnected via a NoC. The Shim tiles feature high bandwidth DMAs to move data in and out of the NPU. The Mem tiles are intermediate large 512 kB512\text{\,}\mathrm{kB} buffers coupled with a DMA engine. The AIE tiles are VLIW processors with a scalar and vector datapath; each AIE core runs an independent program.](https://arxiv.org/html/2607.09385/2607.09385v1/x2.png)

TABLE I: Comparison of commercial NPUs for laptop SoCs.

|  | Intel’s |  |  |  |
| --- | --- | --- | --- | --- |
| AI Boost ([7]) | Qualcomm’s |  |  |  |
| Hexagon ([3]) | Huawei’s |  |  |  |
| Ascend 310 ([5]) | AMD’s |  |  |  |
| XDNA™2 ([21]) |  |  |  |  |
| Open-Source Software Stack | Yes | No | Yes | Yes |
| Spatial Dataflow |  |  |  |  |
| Architecture | No | No | No | Yes |
| Peak Throughput |  |  |  |  |
| (TOPS) | 48 | 45 | 16 | 50 |

The XDNA™ 2 NPU ([19]) adopts a two-dimensional spatial architecture composed of VLIW processing units interconnected via a NoC (Fig. 2). It operates between 1.3 GHz1.3\text{\,}\mathrm{GHz} and 1.8 GHz1.8\text{\,}\mathrm{GHz} depending on power mode.
The architecture is organized into eight columns, each comprising a Shim tile, a Mem tile, and four AIE tiles. The Shim tile connects the NoC to DRAM and provides a high-throughput DMA engine supporting 4-D transfers over two 256-bit channels. The Mem tile integrates 512 kB512\text{\,}\mathrm{kB} of multi-bank interleaved memory and a DMA engine with 4-D transfer support.
Computation is performed on AIE tiles. Each AIE core is a VLIW processor issuing up to seven instructions across scalar and vector datapaths, enabling overlap of control and compute. The vector unit sustains 64 multiply-accumulate (MAC) operations per cycle for bfloat16 inputs with fp32 accumulation. Each tile includes 64 kB64\text{\,}\mathrm{kB} of banked data memory, 16 kB16\text{\,}\mathrm{kB} of program memory, and an on-tile DMA engine supporting 3-D transfers.

## III Related Work

### III-A Commercial NPUs

With the rise of early DNNs for vision tasks such as convolutional neural networks (CNNs) ([18]), specialized accelerators were integrated into edge processors to improve energy efficiency ([15]). A similar trend is now emerging with the integration of NPUs into portable SoCs to accelerate LLMs ([20]). However, commercially available NPUs exhibit significant architectural diversity, making it important to contextualize the design space targeted by STEEL.
A primary comparison metric is peak throughput, typically reported in tera operations per second (TOPS). Table I highlights large variation across platforms, reflecting differences in both scale and target workloads.
The spatial dataflow architecture of XDNA™ 2 (Sec. II-C) differs fundamentally from other NPUs. Its 32 AIE cores execute independent programs and communicate through the NoC, enabling fine-grained, operator-specific mappings across diverse ML workloads. This flexibility expands the mapping design space, increasing the complexity of achieving efficient implementations.
Finally, practical deployment depends heavily on the maturity and accessibility of the software stack. Open-source stacks improve usability for both researchers and developers; Table I shows that Hexagon remains the only platform relying on a closed software stack.

### III-B Attention Mapping on NPUs

Optimizing the mapping of complex ML operators, such as the attention mechanism, for a specific NPU can lead to greatly improved performance compared to a naive approach ([12]). Efficient mappings still heavily rely on expert knowledge and are very time-consuming. Companies usually provide a library of hand-tuned mappings for their custom processors, the most well-known being BLAS ([11]), CUDA, or ROCm. Several attempts have been made to optimize operator libraries ([24]); however, due to the vast differences among hardware platforms, no uniform approach for automatically generating operator libraries has emerged yet.

Optimized mappings of the attention mechanism to NPUs have recently been proposed. FastAttention ([12]) maps attention onto the Ascend 310 NPU. It introduces a multi-level tiling strategy and generates the causal mask on the fly, avoiding the need to store the mask and move it through the memory hierarchy.
STEEL shares some similarities with FastAttention in its masking strategy: we also find that generating the mask directly on the AIE core is more efficient than fetching it from DRAM. However, the Ascend NPU architecture differs substantially. In FastAttention, the AI cores do not communicate directly, and the attention computation is not distributed across multiple cores to form a balanced pipeline, unlike in STEEL. Moreover, FastAttention does not address the workload imbalance induced by the causal mask.
DATO ([6]) presents a task-based programming model for executing several ML operators on the XDNA™ 1 NPU, including attention. Although DATO implements a FlashAttention kernel, it does not discuss pipeline balancing nor the handling of sparsity induced by the attention mask.

## IV Methods

Algorithm 1  STEEL Algorithm

1:procedure First Stage(QQ, KK) →(A,m,m​‘)\to(A,m,m`)

2:  for 0≤i<Tq0\leq i<T\_{q} do

3:   Acquire QiQ\_{i}, mm, m​‘m`

4:   for 0≤j<Tk​v0\leq j<T\_{kv} do

5:     Acquire KjK\_{j}, Ai​jA\_{ij}

6:     Ai​j←matmul\_b\_transposed\_unswizzle​(Qi,Kj)A\_{ij}\leftarrow\text{matmul\\_b\\_transposed\\_unswizzle}(Q\_{i},K\_{j})

7:     Ai​j←scale\_and\_mask​(Ai​j,l​o​g​2​ed)A\_{ij}\leftarrow\text{scale\\_and\\_mask}(A\_{ij},\frac{log2e}{\sqrt{d}})

8:     m​‘←max⁡(m,rowmax​(Ai​j))m`\leftarrow\max(m,\mathrm{rowmax}(A\_{ij}))

9:     Release KjK\_{j}, Ai​jA\_{ij}, mm, m​‘m`

10:   Release QiQ\_{i}

11:procedure Second Stage(AA, mm, m​‘m`) →(P,l,v)\to(P,l,v)

12:  for 0≤i<Tq0\leq i<T\_{q} do

13:   for 0≤j<Tk​v0\leq j<T\_{kv} do

14:     Acquire Ai​jA\_{ij}, Pi​jP\_{ij}, mm, m​‘m`, ll, l​‘l`, vv

15:     Pi​j←eAi​j−m​‘P\_{ij}\leftarrow e^{A\_{ij}-m`}

16:     v←em−m​‘v\leftarrow e^{m-m`}

17:     ℓ←v​ℓ+r​o​w​s​u​m​(Pi​j)\ell\leftarrow v\ell+rowsum(P\_{ij})

18:     m←m​‘​and​l←l​‘m\leftarrow m`\,\text{and}\,l\leftarrow l`

19:     Release Ai​jA\_{ij}, Pi​jP\_{ij}, mm, m​‘m`, ll, vv

20:procedure Third Stage(PP, VV, ll, vv) →(O)\to(O)

21:  for 0≤i<Tq0\leq i<T\_{q} do

22:   Acquire OiO\_{i}

23:   for 0≤j<Tk​v0\leq j<T\_{kv} do

24:     Acquire Pi​jP\_{ij}, VjV\_{j}, ll, vv

25:     Oi←scale\_swizzle​(Oi,v)O\_{i}\leftarrow\text{scale\\_swizzle}(O\_{i},v)

26:     Oi←matmul​(Pi​j,Vj)O\_{i}\leftarrow\text{matmul}(P\_{ij},V\_{j})

27:     Release Pi​jP\_{ij}, VjV\_{j}, θ1\theta\_{1}

28:   Oi←scale​\_​s​w​i​z​z​l​e​(Oi,l−1)O\_{i}\leftarrow\text{scale}\\_swizzle(O\_{i},l^{-1})

29:   Release OiO\_{i}

### IV-A The STEEL Pipeline

We design STEEL starting from a three-stage formulation of FlashAttention-2, where stages compute attention scores Ai​jA\_{ij}, apply online softmax, and update the mm and ℓ\ell statistics while accumulating Pi​j​VjP\_{ij}V\_{j} into OiO\_{i}. Profiling this baseline reveals load imbalance across stages; we iteratively refine the decomposition to obtain a balanced pipeline.

Algorithm 1 shows the final formulation, where each procedure maps to one stage executed on a dedicated AIE core. STEEL is implemented using the IRON Python Bindings API, which maps applications to the XDNA™ 2 NPU and expresses inter-PE communication via ObjectFIFO ([9]). The First Stage uses matmul\_b\_transposed\_unswizzle to compute Ai​jA\_{ij}, while the Third Stage uses matmul to accumulate into OiO\_{i}.

Efficient execution on AIE cores requires tiles to match the vector-unit layout. Layout transformations can be performed either by the vector unit or via DMA. Since matrix multiplication requires 4-D transfers and AIE-local DMA supports only 3-D, we use the Mem tile DMA, which provides full 4-D support.

In the Third Stage, OiO\_{i} is scaled by em−m​’e^{m-m’} and l−1l^{-1} using scale\_swizzle. Each row is scaled independently while stored in a swizzled layout. To maximize vector utilization, we broadcast each scaling factor across 8 elements, concatenate them into a 64-element vector, and apply element-wise multiplication across 8 rows. This fully utilizes the 512 bit512\text{\,}\mathrm{bit} vector registers (64 ×\times 16 bit16\text{\,}\mathrm{bit} elements).

To handle sparsity induced by the causal mask, each AIE tile tracks the coordinates (i,j)(i,j) of tile Ai​jA\_{ij}, enabling three cases: (i) fully masked tiles are skipped, (ii) unmasked tiles are processed normally, and (iii) partially masked tiles apply the mask locally in the Second Stage. This approach generalizes to other masking schemes, such as windowed attention in time-series models.

### IV-B Macroscale Data Movement

Subsection IV-A describes the intra-pipeline algorithm and data movement; mapping STEEL to hardware additionally requires placing pipelines on the PE array and orchestrating data transfers between DRAM and the NPU. Figure 3 shows the pipeline placement and the movement of QQ, KK, and VV tiles. We use the IRON distribute primitive to assign one QQ tile per pipeline, join to collect OO tiles, and broadcast KK and VV to all pipelines.

Memory port availability is a key constraint on XDNA™ 2. Each Mem tile provides six ports, limiting the number of concurrent ObjectFIFOs per tile. With 10 pipelines, we therefore use two Mem tiles to distribute each wave of 10 QQ tiles.
To swizzle the Pi​jP\_{ij} tiles between the second and third stages of the STEEL pipeline, we use the Mem tile DMA engine. Consequently, each STEEL pipeline consumes four Mem tile ports: one for QQ, one for OO, and two for swizzling PP. In addition, the pipelines collectively share two Mem tile ports for broadcasting KK and VV. Overall, the 10 STEEL pipelines use 42 Mem tile ports out of 48.

### IV-C Sparsity-Aware Pipeline Placement

Broadcasting KK and VV is required to stay within the Mem tile port budget (Subsection IV-B), but introduces a synchronization constraint: a broadcast can start only when all consumers are ready. While this is trivial under uniform workloads, the causal mask in language models (LMs) creates load imbalance across STEEL pipelines. Although double buffering mitigates this effect, we find that explicitly balancing sparsity across pipeline groups significantly improves runtime.

Figure 4 compares two placement strategies over the attention matrix AA. Uniform placement assigns contiguous TqT\_{q} chunks to pipelines, resulting in uneven sparsity within a group. We instead propose a sparsity-aware placement that distributes consecutive rows across pipelines to equalize sparsity, yielding a 38,% speedup.

![Figure 3: Overview of the data movement between DRAM and the STEEL pipelines. A chunk of 5 QQ tiles is brought down to the Mem tile, and then dispatched between 5 STEEL pipelines. Tiles of KK and VV are broadcast to each STEEL pipeline. Once a complete head of KK and VV has been broadcast, output tiles are sent back through the leftover Mem and Shim tiles.](https://arxiv.org/html/2607.09385/2607.09385v1/x3.png)

![Figure 4: Description of two pipeline placement strategies over the attention matrix AA. On the left side, the STEEL pipelines are placed uniformly on AA, resulting in an imbalance of the sparsity amount in the pipeline group and causing stalls. On the right side, we show our sparsity-aware pipeline placement where the sparsity between STEEL pipelines within a group is close.](https://arxiv.org/html/2607.09385/2607.09385v1/x4.png)

## V Results

In this section, we describe our evaluation setup and present an extensive benchmark of STEEL. We begin by comparing STEEL against a standard layer-by-layer attention implementation on XDNA™ 1. We then benchmark STEEL against the SotA FlashAttention implementation provided by DATO ([6]), also on XDNA™ 1. Finally, in subsection V-D, we benchmark the fused attention across the processing units available in the AMD Ryzen™ AI 9 HX 370 SoC. AMD Ryzen™ AI 9 HX 370 integrates 16 Zen5 CPUs, an RDNA 3.5 GPU, and the XDNA™ 2 NPU.

### V-A Evaluation Setup

Results from subsections V-B and V-D were collected on the AMD Ryzen™ AI 9 HX 370 SoC, a TSMC 4 nm4\text{\,}\mathrm{nm} single-die platform with 12 Zen5 cores (24 threads) up to 5.1 GHz5.1\text{\,}\mathrm{GHz}. The RDNA 3.5 GPU comprises 16 compute units (CUs) (1024 shaders at 2900 MHz2900\text{\,}\mathrm{MHz}, GFX1150 ISA) and shares 32 GB32\text{\,}\mathrm{GB} DRAM with the CPU and NPU. The system also integrates the XDNA™ 2 NPU (see Subsection II-C).

CPU and GPU benchmarks use the TorchLib C++ frontend from PyTorch 2.1 ([16]), leveraging ROCm 6.4 and the HIP runtime, with scaled\_dot\_product\_attention executed via the FlashAttention backend. The NPU is configured in turbo mode using XRT (1.8 GHz1.8\text{\,}\mathrm{GHz}). Latency results are averaged over 50 iterations after 25 warm-ups.
Power is measured using AMD’s AGT tool at a 50 ms50\text{\,}\mathrm{ms} sampling rate over 100 iterations, reporting average consumption.

### V-B Attention Implementation Benchmark

![Figure 5: Benchmark of the layer-by-layer attention against the STEEL’s fused-attention on XDNA™ 1. The attention configuration is from BERT, with 12 heads and a head dimension of 64.](https://arxiv.org/html/2607.09385/2607.09385v1/Figures/Phoenix-STEEL-IRON-benchmark.png)

Figure 5 compares STEEL with a layer-by-layer attention implementation built from IRON operators. STEEL achieves a 22.8×\times speedup on average, demonstrating the benefit of fused attention. This gain stems from reduced overhead: STEEL loads a single design onto the NPU, whereas the baseline requires separate GEMM, Softmax, and Scale kernels, incurring context-switch costs. Additionally, the baseline transfers intermediate tensors AA and PP between the NPU and DRAM, while STEEL avoids this by not materializing them off-chip.

Figure 6 compares off-chip data movement for layer-by-layer (standard) and fused attention across sequence lengths. While individual IRON operators are optimized for locality, fusion further improves it. The model accounts for the IRON GEMM dataflow and STEEL’s dataflow to estimate transfer volume. Blue curves indicate the theoretical lower bound; in practice, limited on-chip buffering leads to higher realized traffic.
Figure 6 also shows how off-chip traffic scales with core count. Multi-core execution on the full XDNA™ 2 NPU reduces transfers due to increased on-chip buffering and broadcast reuse. For a sequence length of 4096, STEEL achieves a 19.4×\times reduction, from 9.7 GB9.7\text{\,}\mathrm{GB} to 0.5 GB0.5\text{\,}\mathrm{GB}.

### V-C Comparison with State-of-the-Art on NPU

![Figure 6: Off-chip transfer volume required to execute attention on XDNA™ 2 across several implementations as a function of sequence length. Dashed curves denote standard layer-by-layer attention implementations, whereas solid curves correspond to STEEL. Blue curves indicate the theoretical lower bound for each implementation and assume an infinite on-chip buffer capacity. Single-core uses one AIE core, while multi-core uses the full XDNA™ 2 array.](https://arxiv.org/html/2607.09385/2607.09385v1/Figures/memory-transfer-model.png)

To the best of the authors’ knowledge, DATO ([6]) is the only published work that reports FlashAttention latency on XDNA™ NPUs. However, DATO targets the XDNA™ 1 NPU. To show that STEEL’s performance is not specific to a single NPU generation, we port STEEL to XDNA™ 1.
Whereas XDNA™ 2 provides eight columns of PEs, XDNA™ 1 provides five. Accordingly, we deploy one STEEL pipeline per column on the first three AIE cores. We then place an additional STEEL pipeline on the last AIE core of each of the first three columns, as illustrated in Figure 3. Finally, XDNA™ 1 AIE cores do not include a dedicated exponentiation unit; instead, we implement exponentiation via a lookup table.
We benchmark against DATO in an application-relevant setting by selecting the BERT attention configuration, which uses 12 heads with a head dimension of 64. In Figure 7, we observe that STEEL consistently outperforms DATO across all sequence lengths. On average, STEEL accelerates attention by 9.6×\times relative to DATO.
This speedup primarily stems from how DATO partitions FlashAttention into four stages, which introduces a fundamental load imbalance in the pipeline. In particular, DATO performs output rescaling in the final stage. This rescaling is an element-wise multiplication between the precomputed factor emj−1−mje^{m\_{j-1}-m\_{j}} and the current output tile OjO\_{j}. Consequently, the fourth stage performs only Bq⋅Bk​vB\_{q}\cdot B\_{kv} MACs, which is substantially less than the Bq⋅Bk​v⋅dB\_{q}\cdot B\_{kv}\cdot d work required for the GEMM in the first stage. For the BERT head dimension d=64d=64, the last stage therefore performs 64×\times less computation than the first stage, resulting in a suboptimal mapping.
Additionally, we were unable to benchmark DATO for sequence lengths greater than 4096 due to what appears to be an exponential increase in compilation time.

### V-D Attention Benchmark on AMD Ryzen™ AI 9 HX 370

![Figure 7: Benchmark of STEEL against DATO [6] on XDNA™ 1 for several sequence lengths. The attention configuration is from BERT, with 12 heads and a head dimension of 64.](https://arxiv.org/html/2607.09385/2607.09385v1/Figures/Phoenix-STEEL-DATO-benchmark.png)

AMD Ryzen™ AI 9 HX 370 is a heterogeneous SoC in which the XDNA™ 2 NPU is the preferred compute engine for low-power DNN inference.
To verify that the NPU is the most appropriate engine for attention inference, we benchmark optimized attention kernels across the three compute engines available in AMD Ryzen™ AI 9 HX 370, as shown in Figure 8: the XDNA™ 2 NPU, the RDNA 3.5 GPU, and the Zen5 CPU.
To emulate an application-representative workload, we adopt the attention dimensions of Llama3.1-1B ([8]). In this model, attention uses 32 heads with a head dimension of 64, and the maximum supported context length is 128k tokens. We sweep the sequence length from 2048 to 32768 to capture a common usage regime in which the model processes substantial contextual information (e.g., a codebase).
On average, STEEL reduces energy consumption by 9.17×\times and 1.75×\times relative to the CPU and GPU, respectively.
We further observe that STEEL’s energy-efficiency advantage over both the CPU and the GPU grows with sequence length, most notably between 2048 and 8192. This trend shows that the STEEL pipeline reaches a warmer steady state and attains near-peak throughput at approximately a sequence length of 8192.

## VI Conclusion

We presented STEEL, a dataflow formulation of the FlashAttention algorithm targeting the XDNA™ NPU family. STEEL carefully balances the workload across a three-stage pipeline of AIE cores and uses a sparsity-aware pipeline placement to mitigate the workload distribution imbalance induced by causal masking.
On XDNA™ 1, STEEL outperforms the previous SotA implementation of flash-attention on XDNA™ 1 ([6]) by reducing the latency by 9.6 ×\times on average.
On the AMD Ryzen™ AI 9 HX 370 SoC, STEEL reduces energy consumption by 9.17×\times and 1.75×\times relative to the CPU and GPU, respectively.

![Figure 8: Benchmark of the attention energy efficiency on AMD Ryzen™ AI 9 HX 370 for various sequence lengths. The attention’s configuration is from Llama3.1-1B [8] with 32 heads and a head dimension of 64.](https://arxiv.org/html/2607.09385/2607.09385v1/Figures/Strix-NPU-GPU-CPU-benchmark.png)

## Acknowledgment

This work has received funding from the Swiss State Secretariat for Education, Research, and Innovation (SERI) under the SwissChips initiative.

