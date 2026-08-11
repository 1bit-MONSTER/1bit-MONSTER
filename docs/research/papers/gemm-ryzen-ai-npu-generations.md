# Striking the Balance: GEMM Performance Optimization Across Generations of Ryzen AI NPUs

arXiv: 2512.13282

## Abstract
The high computational and memory demands of modern deep learning (DL) workloads have led to the development of specialized hardware devices from cloud to edge, such as AMD's Ryzen AI XDNA NPUs. Optimizing general matrix multiplication (GEMM) algorithms for these architectures is critical for improving DL workload performance. To this end, this paper presents a common systematic methodology to optimize GEMM workloads across the two current NPU generations, namely XDNA and XDNA2. Our implementations exploit the unique architectural features of AMD's NPUs and address key performance bottlenecks at the system level. End-to-end performance evaluation across various GEMM sizes demonstrates state-of-the-art throughput of up to 6.76 TOPS (XDNA) and 38.05 TOPS (XDNA2) for 8-bit integer (int8) precision. Similarly, for brain floating-point (bf16) precision, our GEMM implementations attain up to 3.14 TOPS (XDNA) and 14.71 TOPS (XDNA2). This work provides significant insights into key performance aspects of optimizing GEMM workloads on Ryzen AI NPUs.

## 1. Introduction

The widespread adoption of deep learning (DL) applications, combined with their intensive computational requirements, has driven the emergence of specialized hardware platforms.
While cloud data centers continue to provide large-scale acceleration ((Nvidia, 2023; AMD, 2025g; N. P. Jouppi, D. Hyun Yoon, M. Ashcraft, M. Gottscho, T. B. Jablin, G. Kurian, J. Laudon, S. Li, P. Ma, X. Ma, T. Norrie, N. Patil, S. Prasad, C. Young, Z. Zhou, and D. Patterson, 2021; S. Xu and C. Ramakrishnan, 2024; J. Coburn, C. Tang, S. A. Asal, N. Agrawal, R. Chinta, H. Dixit, B. Dodds, S. Dwarakapuram, A. Firoozshahian, C. Gao, K. Gondkar, T. Graf, J. Hu, J. Huang, S. Hughes, A. Hutchin, B. Jakka, G. J. Chen, I. Kalyanaraman, A. Kamath, P. Kansal, E. Kazi, R. Levenstein, M. Maddury, A. Mastro, S. Medaiyese, P. Modi, J. Montgomery, S. Nadathur, A. Nagpal, A. Narasimha, M. Naumov, E. Ozer, J. Park, P. Ramani, H. Reddy, D. Reiss, D. Roy, S. Sekar, A. Sharma, P. Shetty, A. Sukumaran-Rajam, E. Tal, M. Tsai, S. Varshini, R. Wareing, O. Wu, X. Xie, J. Yang, H. Yu, T. Zargar, Z. Zeng, F. Zhang, A. Matthews, X. Jiao, J. Zhang, E. Menage, T. E. Stokke, and M. Sourouri, 2025; 37)), the increasing demand for energy-efficiency, low-latency, as well as enhanced privacy and security
has motivated the integration of DL accelerators on the edge ((Rico et al., 2024; Knopp et al., 2024; Intel, 2025; Mahurin, 2023; Nvidia, 2022)).
To this end, AMD released the Ryzen™ AI processors ((Rico et al., 2024)), featuring multi-core CPUs, an integrated GPU, and a new neural processing unit (NPU).
The NPU features the XDNA™ architecture and serves as a dedicated DL accelerator integrated with x86 processors.
The AMD NPU is an evolution of the AI Engine (AIE) architecture, utilized in Versal™ adaptive system-on-chip (SoC) platforms ((Ahmad et al., 2019; Freeman et al., 2021)) and Alveo™ V70 accelerator cards ((AMD, 2023)).
The NPU architecture leverages the characteristics of modern DL workloads, where compilers can determine most of the control flow statically.
In particular, AMD NPUs provide an explicit data movement architecture, which both reduces hardware complexity and enables high performance, while offering substantially higher energy-efficiency compared to dynamically scheduled architectures, such as CPUs and GPUs ((Rico et al., 2024)).

As modern DL workloads are dominated by general matrix multiplication (GEMM) operations, several prior works have aimed to design and optimize GEMM workloads on the Versal platforms
((Taka et al., 2023; Zhuang et al., 2023a, b; Taka et al., 2024; Zhuang et al., 2024; Deng et al., 2024; Zhuang et al., 2025; Wang et al., 2025; Mhatre et al., 2025)).
The Versal SoC devices integrate an FPGA fabric, which acts as another level of memory hierarchy and is utilized for efficient exploitation of data reuse
in GEMM.
Furthermore, the flexibility of the FPGA allows the design of customized tiling schemes ((Taka et al., 2024; Zhuang et al., 2023a)) and tailored data layout transformations ((Wang et al., 2025)).

In this work, we conduct a comprehensive study to optimize GEMM workloads on Ryzen AI NPUs.
Due to the absence of FPGA fabric, GEMM mapping and optimization on the AMD NPUs is a fundamentally different problem compared to Versal devices.
This highlights the necessity for a separate methodology to address several new design challenges introduced by the NPUs.
Having identified this important research gap, we propose a unified systematic framework to maximize GEMM performance across the two current generations, namely XDNA and XDNA2.
We extensively focus on the end-to-end GEMM performance
and introduce a novel procedure to enable system-level optimization.
Within the framework, matrices are retained in regular order (row- and column-major) in main memory (DRAM).
This facilitates seamless integration with tensor libraries for DL, such as GGML ((GGML, 2025)), while also enabling the implementation of high-performance GEMM libraries, similar to GPUs ((Nvidia, 2025a, b; AMD, 2025l, p)).
Moreover, our GEMM designs efficiently leverage several architectural features of the NPUs (*e.g.,* on-the-fly tensor transformations), while also addressing critical performance bottlenecks (*e.g.,* via sophisticated data movement design between the NPU and DRAM). The key contributions of this paper are:

* •

A systematic methodology to optimize GEMM workloads on Ryzen AI NPUs through analytical modeling along with hardware profiling.
Our methodology is general in scope; in this paper we apply it across the two current NPU generations. Observing the inverse relationship between compute and off-chip memory, our approach is based on identifying the balanced point that maximizes performance.
* •

A sophisticated GEMM implementation that enables sufficient contiguous DRAM accesses to maximize performance.
Our design leverages the multi-dimensional tensor addressing feature across the entire NPU hierarchy to transform matrices into the tiled layout expected by NPU cores, enabling matrices to remain in standard layout in DRAM (*i.e.,* row- and column- major order) without explicit pre-tiling.
* •

A thorough experimental evaluation on two mini PCs, each incorporating an NPU from a different generation, which exhibits GEMM performance up to 6.76 TOPS (XDNA) and 38.05 TOPS (XDNA2) for 8-bit integer (int8), as well as 3.14 TOPS (XDNA) and 14.71 TOPS (XDNA2) for brain floating-point (bf16).
Furthermore, we present roofline sweeps to provide a holistic view of performance across hundreds of GEMM sizes for each data type.
* •

We provide essential insights regarding key performance considerations for GEMM optimization on AMD’s NPU architecture.

## 2. Related Work

Having identified the importance of GEMM in DL, several prior works have proposed frameworks that aim to optimize GEMM workloads on Versal devices.
Some prior works ((Zhuang et al., 2023a, b, 2024, 2025; Wang et al., 2025)) explore end-to-end GEMM performance including DRAM, while others ((Taka et al., 2023, 2024; Deng et al., 2024; Mhatre et al., 2025)) focus exclusively on the AIE computation part (*i.e.,* excluding off-chip considerations).
One common design choice across all of these works is to partition GEMM across the reduction dimension, driven by the limited number of ports in the AIE-FPGA interface.
For instance, MaxEVA ((Taka et al., 2023, 2024)) utilizes 20% of the AIE cores to perform adder-tree reduction, limiting the GEMM compute efficiency to 80%.
In a similar manner, GAMA ((Mhatre et al., 2025)) exploits the cascade interface to transfer partial accumulations across AIE cores, observing an average performance degradation of 7% due to cascade stalls.
In contrast, as we demonstrate in this work, all cores on AMD NPUs can perform GEMM computation in an independent fashion, thereby maximizing the compute efficiency.

Some prior works have also explored mapping GEMM on AMD NPUs. In ((Rösti and Franz, 2025)), the authors explore fine-tuning GPT-2 on the Ryzen AI processors.
Specifically, they offload the time-intensive GEMM computation on XDNA and attain 2.8×\times speedup compared to a CPU-only implementation.
Furthermore, they utilize a GEMM implementation similar to the non-optimized programming example in ((AMD, 2025n)).
In ((Fang et al., 2025)), a task-based programming model for dataflow accelerators is presented, demonstrating a GEMM implementation on XDNA.
In particular, they attain up to 5.04 TOPS and 1.95 TOPS for int8 and bf16 data types, respectively, which closely matches the performance in ((AMD, 2025n)).
In contrast, our optimized GEMM designs achieve superior performance of up to 6.76 TOPS (34% higher) and 3.14 TOPS (62% higher) on XDNA for int8 and bf16, respectively.

Related workloads were covered in prior publications.
The authors in ((Zhuang et al., 2025)) map a ResNet layer on XDNA, utilizing only 20% of the XDNA cores.
In ((Deshmukh et al., 2025)), a compiler framework for dynamic attention folding on AMD NPUs is presented.
Moreover, in ((Karami et al., 2025)) the authors perform characterization of generative AI workloads on Ryzen AI processors.
Finally, other works leverage the NPUs to perform stencil applications ((Xu et al., 2025)), accelerate Fortran intrinsics ((Brown and Rodriguez-Canal, 2025)), and implement a variant of the fast Fourier transform (FFT) ((Nozaki et al., 2024)).

## 3. NPU Architecture, Features & Programming

![Figure 1. Architecture of Ryzen AI NPUs.](https://arxiv.org/html/2512.13282/2512.13282v1/x1.png)

### 3.1. NPU Architecture Overview

The architecture of both XDNA and XDNA2 is illustrated in Fig. 1.
The NPU is a modular and scalable
architecture, comprising a 2D array of identical compute tiles (CompTiles) ((Rico et al., 2024; AMD, 2025t)).
CompTiles include the processing cores which operate out of a local memory (L1).
The NPU core is a very long instruction word (VLIW) processor with
single instruction multiple data (SIMD) datapath, supporting both fixed-point and floating-point operations.
NPUs incorporate a second level of on-chip memory (L2) via the memory tiles (MemTiles).
These are arranged in a single row, located below the array of CompTiles (Fig. 1).
Finally, NPUs include a last row of interface tiles (ShimTiles) to provide communication with DRAM.

Data movement between the levels of memory hierarchy is facilitated by direct memory access (DMA) engines, which are integrated across all NPU tiles.
DMAs move data between the NPU tiles by utilizing the configurable interconnects (switches), shown as black squares in Fig. 1.
The DMA engines of ShimTiles read/write data to DRAM via the NPU network-on-chip (NoC) and SoC-level fabric of the Ryzen AI chips ((Rico et al., 2024; Subramon et al., 2023)).
Task scheduling and data movement are orchestrated by an on-chip command processor.
This processor controls the data movement between ShimTiles and DRAM at runtime, allowing the NPU to focus exclusively on the main computation.
The command processor is also responsible for (re-)configuring the NPU compute kernels, switches, and DMA transfers.

The XDNA has 20 compute cores, which are organized as a 4×\times5 array (rows ×\times columns) of CompTiles ((Rico et al., 2024)), while the XDNA2 contains 32 cores as a 4×\times8 array ((AMD, 2024)).
Both NPU generations have 64 KB of memory per L1 tile and 512 KB of memory per L2 tile ((Rico et al., 2024; AMD, 2025t; Deshmukh et al., 2025)).
XDNA and XDNA2 natively support int8, int16 and bf16 precisions.
Additionally, XDNA2 has hardware support for block floating-point (bfp16) datatype ((AMD, 2025a)), where a block of eight numbers shares one common exponent ((AMD, 2025k; Darvish Rouhani et al., 2020)).
XDNA2 offers increased theoretical peak compute capabilities, delivering up to 50 TOPS ((AMD, 2025f, 2024)), compared to the 10 TOPS of XDNA ((Rico et al., 2024)).

### 3.2. Data Movement Architecture

![(a) ](https://arxiv.org/html/2512.13282/2512.13282v1/x2.png)

Ryzen AI NPUs incorporate a dedicated data movement architecture that allows programmers to explicitly configure data transfers between all levels of the memory hierarchy.
Data moves from a source DMA channel to a destination DMA channel through a circuit- or packet-switched stream, using one or more configurable switches between them.
The source channel is a memory-map to stream (MM2S) channel which reads data from memory and pushes it to the stream switches.
Conversely, the destination channel receives data from the stream switches and writes it to the memory-mapped space (S2MM).
Each CompTile and ShimTile includes two MM2S and two S2MM DMA channels, while MemTiles incorporate six MM2S and six S2MM channels ((AMD, 2025t)).

Fig. 2a shows an example of moving an input buffer AA from DRAM to an L2 MemTile, and eventually to L1, where it can be utilized by the corresponding core.
In a similar fashion, Fig. 2b depicts the data movement of an output buffer from L1 to L2, and eventually to DRAM.
The synchronization of data buffers between the DMAs and the corresponding module (*e.g.,* the NPU core or DRAM) is managed by hardware lock units ((AMD, 2025t; Rico et al., 2024)).

DMAs run independently and in parallel with computation on cores.
They are programmed by configuring a sequence of buffer descriptors (BDs).
A BD contains all the required information associated with a specific DMA transfer, such as the amount of data to read/write, the memory addresses involved, and the locks to acquire/release before and after each transfer ((AMD, 2025t, d)).
BDs support both linear memory addressing and multi-dimensional address generation, enabling on-the-fly data layout transformations required for DL tensors.
CompTiles and ShimTiles support each 3D tensor addressing, while MemTiles incorporate 4D addressing.
This important DMA addressing feature is extensively exploited by our GEMM implementation across all tiles in the NPU architecture (refer to Sec. 4.3), allowing tensors to be stored in regular order in DRAM.

### 3.3. Programming Tools

In this work, we use IRON, an open-source close-to-metal toolchain for developing programs on Ryzen AI NPUs ((AMD, 2025m)).
As a low-level toolkit, IRON enables fine-grained control of the NPU architectural attributes, such as explicit data movement and complex access patterns supported in DMAs, while also providing convenient programming abstractions ((Hunhoff et al., 2025)).
These characteristics render IRON a compelling choice for GEMM implementation, where explicit control of the NPU architectural features is critical to performance.

![Figure 3. Proposed GEMM multi-level tiling scheme (a), and GEMM mapping strategy on XDNA (b) and XDNA2 (c).](https://arxiv.org/html/2512.13282/2512.13282v1/x4.png)

IRON is based on a multi-level intermediate representation (MLIR) ((Lattner et al., 2021)) dialect, named “AIE”, and offers a user-friendly interface to this dialect through Python bindings ((MLIR, 2025)).
Hence, Python scripts can be used to generate code that describes both the data movement across the NPU hierarchy and the compute kernels that run on the cores.
The single-core compute kernels can be written in high-level C++ ((AMD, 2025a)), or low-level SIMD intrinsics ((AMD, 2025c)).
Finally, two options are available for single-core kernel compilation: the proprietary *xchesscc* compiler ((AMD, 2025d, m)) and the open-source *Peano* tool ((AMD, 2025e)).

## 4. GEMM Design & Optimization

### 4.1. Multi-Level Tiling Method

Fig. 3a illustrates the multi-level tiling scheme we use to partition the input and output matrices.
The inner-most (first) tiling level is defined by the supported shapes of the AIE API library for the single-core GEMM kernel ((AMD, 2025a)), and is expressed via the parameters rr ×\times ss ×\times tt.
The AIE API provides multiple optimized modes for each supported precision and enables portability across NPU generations.
The second tiling level is determined by the GEMM kernel that is supported out of local L1 memory, denoted as mctm\_{\text{ct}} ×\times kctk\_{\text{ct}} ×\times nctn\_{\text{ct}}.
Since the NPU is a multi-core architecture, we partition GEMM across multiple cores to exploit spatial-level parallelism and maximize performance.
The third tiling level achieves this partitioning and corresponds to the GEMM size operating on the entire NPU array (parameters and mapping delineated in the next section).
Finally, the outer-most (fourth) tiling level is dictated by the final GEMM sizes of input matrices AA (typically activations) and BB (typically weights), and output matrix CC, expressed as MM ×\times KK ×\times NN.

### 4.2. GEMM Mapping Strategy on NPUs

#### 4.2.1. NPU Array Mapping & Core Design

Our mapping strategy is to parallelize GEMM in space across the MM and NN dimensions, while reduction across the KK dimension is performed in time.
In this manner, all NPU cores perform the same GEMM computation on different data and operate independently.
This leads to maximized performance, since no data communication occurs between the cores, as opposed to Versal devices, where the reduction dimension is partitioned across multiple cores (*e.g.,* ((Taka et al., 2023; Mhatre et al., 2025))).
By using the broadcast feature of the NPU architecture, we can attain spatial parallelization and exploit the inherent data reuse of the GEMM algorithm.
This parallelization is expressed via the number of single-core tiles in the MM and NN dimensions, and is defined by the parameters mrowsm\_{\text{rows}} and ncolsn\_{\text{cols}}, respectively (Fig 3a).
In this work, we map the parameters mrowsm\_{\text{rows}} and ncolsn\_{\text{cols}} in a straightforward fashion to the number of rows and columns of the two NPU architectures, respectively.
In particular, we broadcast each input AA tile across one row in the NPU array (*e.g.,* A​0A0 in Fig. 3b and Fig. 3c for XDNA and XDNA2, respectively).
Similarly, each BB tile is broadcast across one column of NPU cores.
Note that due to the absence of a ShimTile in the last column of XDNA, we choose to map GEMM across 4 rows and 4 columns for this architecture, similar to ((AMD, 2025n; Rösti and Franz, 2025; Fang et al., 2025)).
This enables a symmetric 4 ×\times 4 (mrowsm\_{\text{rows}} ×\times ncolsn\_{\text{cols}}) GEMM implementation, as depicted in Fig. 3b.
For XDNA2, we utilize the entire 4 ×\times 8 array, which results in an asymmetric mapping (Fig. 3c).

Each core performs a GEMM of mctm\_{\text{ct}} ×\times kctk\_{\text{ct}} ×\times nctn\_{\text{ct}} size.
To handle the reduction across KK, the single-core kernel also loads previous partial results of mctm\_{\text{ct}} ×\times nctn\_{\text{ct}} size, performs accumulation and stores the updated results back to the output tiles.
For example, the upper-left core in Fig. 3b, sequentially loads tiles A​0A0 and B​0B0, A​0′A0^{\prime} and B​0′B0^{\prime}, etc., in order to perform reduction.
In this fashion, the output CC tiles remain stationary in the L1 memory of each core (output stationary mapping).
When all tiles complete their accumulation (KK/kctk\_{\text{ct}} tiles in total), the output tile is transferred to an L2 MemTile, and eventually to DRAM. The MemTile design is detailed in Sec. 4.2.2.
Subsequently, a fast vectorized kernel initializes the output CC tile to zero, preparing it for the next accumulation in GEMM tiling.

To overlap GEMM computation with DMA transfers of matrix tiles, we employ double-buffering on both L1 and L2 for the input tiles AA and BB.
However, we retain the output CC tiles as a single buffer on each core.
Since the output tiles are transferred only once for each complete reduction, when KK/kctk\_{\text{ct}} is sufficiently large, the infrequent added latency of a single-buffered output transfer becomes negligible.
At the same time, this design choice frees up valuable L1 memory, thereby enabling higher flexibility in tiling parameter optimization.
The larger tile sizes enabled by this additional memory ultimately lead to higher end-to-end GEMM performance in the general case (refer to Sec. 5.3.2).

#### 4.2.2. Memory Tile Design

The input tiles are temporarily stored in L2 MemTiles before being broadcast to each NPU core.
Note that we load multiple input tiles across the reduction dimension KK into MemTiles (*e.g.,* A​0A0 and A​0′A0^{\prime} in Fig. 3b and 3c).
This is an important aspect of the mapping;
loading multiple tiles allows us to access a larger amount of contiguous data from DRAM, where matrices are stored in regular order (row- and column-major).
Such long contiguous reads result in increased DRAM bandwidth (BW) utilization, which is essential for maximizing
GEMM performance at the system level.
In this work, we retain matrices AA and CC in row-major order, while matrix BB is either in row- or column-major order.
To this end, we introduce another parameter, kmtk\_{\text{mt}}, which specifies the size of the tiles loaded into L2.
In particular, for matrix AA, each L2 MemTile loads a tile of mctm\_{\text{ct}} ×\times kmtk\_{\text{mt}} size.
Similarly, when BB is in column-major order, each L2 MemTile loads a tile of kmtk\_{\text{mt}} ×\times nctn\_{\text{ct}} size.
Storing matrix AA in row-major and BB in column-major provides sufficient contiguous DRAM access for both matrices, which leads to higher GEMM performance (Sec. 5.2.3).
However, when BB is in row-major, MemTiles load the same tile as CompTiles (*i.e.,* kctk\_{\text{ct}} ×\times nctn\_{\text{ct}}), since contiguous
data are accessible across the nctn\_{\text{ct}} dimension.

Due to the 4 ×\times 4 symmetric design of XDNA, each MemTile holds the same amount of input tiles, as illustrated in Fig. 3b.
Specifically, each MemTile broadcasts BB tiles within its own column.
The AA tiles are broadcast across the four rows; we map them in a regular fashion to the four MemTiles: the MemTile in column 0 holds A​0A0 (which will be broadcast across row 0), the MemTile in column 1 holds A​1A1 (which will be broadcast across row 1), etc.
In contrast, for XDNA2, we map the four AA tiles to eight MemTiles in an alternating pattern across even columns due to its 4 ×\times 8 asymmetric design.
The mapping for BB remains the same.
In this design, even MemTiles hold more tiles than their odd counterparts, as depicted in the logical view of Fig. 3c.
This particular mapping aids the IRON tool to leverage the NPU architectural feature of directly accessing the memory of the neighboring MemTile ((AMD, 2025t)).
Therefore, when buffer sizes exceed the capacity of a specific MemTile, IRON physically allocates buffers to a neighboring MemTile.

Observe that four output CC tiles are aggregated across each column into a MemTile (Fig. 3b and 3c).
This is because four CC tiles need to be transferred concurrently, while ShimTiles provide only two S2MM channels.
Hence, we exploit the six S2MM channels available in each MemTile ((AMD, 2025t)) to temporarily store the four output tiles, before they are transferred to DRAM via ShimTiles.

We define the native GEMM size, as (mctm\_{\text{ct}} ⋅\cdot mrowsm\_{\text{rows}}) ×\times kmtk\_{\text{mt}} ×\times (nctn\_{\text{ct}} ⋅\cdot ncolsn\_{\text{cols}}).
This corresponds to the GEMM size that operates natively on the entire NPU array, while also ensuring high performance.
Moreover, note that although we arbitrarily map matrix AA across rows and matrix BB across columns, the reverse mapping is equally feasible, yielding symmetrical solutions across the MM and NN dimensions.
Finally, although an input stationary mapping can also be employed, it would not be adequate to efficiently support arbitrary GEMM dimensions.
Specifically, partial results would need to be temporarily stored in MemTiles, and subsequently reloaded to CompTiles for exploitation of data reuse in GEMM.
This would require three input channels for efficient GEMM computation, while CompTiles provide two inputs channels.

### 4.3. On-The-Fly Tensor Transformations

We extensively exploit the multi-dimensional addressing feature of DMAs
to reorganize data into tiled layouts, as needed by the NPU cores.
This enables matrices to be stored in standard order in DRAM (row- and column-major).
The single-core GEMM kernels assume that matrices are pre-tiled ((AMD, 2025a)).
In particular, the kernels running on each core expect r×s×tr\times s\times t-sized tiles and both data within tiles as well as the tiles themselves to be in row-major order, as illustrated in the upper part of Fig. 4, for the case of matrix AA.
The aforementioned distribution of tiles across multiple cores, along with the requirement for contiguous DRAM accesses necessitates multiple data layout transformations, as explained below.

![Figure 4. On-the-fly DMA transformations for matrix A.](https://arxiv.org/html/2512.13282/2512.13282v1/x5.png)

As matrices are transferred from DRAM to NPU, DMAs apply a series of transformations depending on the addressing capabilities of each NPU tile.
Fig. 4 shows the transformations of each DMA channel associated with the transfer of matrix AA tiles (see Fig. 2a for DMA channels).
Initially, a ShimTile MM2S channel reads a tile of mctm\_{\text{ct}} ×\times KK size from DRAM.
Since ShimTiles support 3D addressing, the row-major mctm\_{\text{ct}} ×\times KK tile is transformed into multiple smaller mctm\_{\text{ct}} ×\times kmtk\_{\text{mt}} tiles via the MM2S channel (parameters: mctm\_{\text{ct}}, kmtk\_{\text{mt}}, KK).
Before each mctm\_{\text{ct}} ×\times kmtk\_{\text{mt}} tile is stored in the MemTile, another 3D transformation occurs at the S2MM MemTile channel, partitioning it into several mctm\_{\text{ct}} ×\times kctk\_{\text{ct}} tiles (parameters: mctm\_{\text{ct}}, kctk\_{\text{ct}}, kmtk\_{\text{mt}}).

Each MemTile holds a mctm\_{\text{ct}} ×\times kmtk\_{\text{mt}} tile, which it sequentially transmits to the corresponding CompTiles as a series of smaller mctm\_{\text{ct}} ×\times kctk\_{\text{ct}} tiles.
Since the CompTiles expect pre-tiled data, the data layout of each of the smaller tiles requires transformation.
Describing this transfer requires five parameters, namely rr, ss, mctm\_{\text{ct}}, kctk\_{\text{ct}}, kmtk\_{\text{mt}}.
However, MemTiles only support 4D addressing.
To circumvent this, we decompose the transformation into two separate transformations, by utilizing the hardware’s data layout transformation features in both the MM2S MemTile (output) channel and the S2MM CompTile (input) channel.
First, the MM2S MemTile channel partitions data into several mctm\_{\text{ct}} ×\times ss tiles, as depicted in Fig. 4 (parameters: ss, mctm\_{\text{ct}}, kctk\_{\text{ct}}, kmtk\_{\text{mt}}).
This enables address linearization within the rr ×\times ss tile, thereby allowing a subsequent 3D transformation in the S2MM CompTile channel to reorganize data into the required layout
(effective parameters: r⋅sr\cdot s, mctm\_{\text{ct}}, kctk\_{\text{ct}}).

Address generation in DMAs occurs at 32-bit granularity ((AMD, 2025t, d)).
DMAs alone cannot perform layout transformations at smaller-precision data types (e.g., individual elements of int8 or bf16 data types).
However, shuffle instructions running on the AIE cores can be utilized to support this operation and enable the fine-grained data swizzling.
In our application, this becomes relevant in use cases where an int8 or bf16 matrix BB is stored in column-major order in DRAM.
To this end, we modify the GEMM kernel to utilize shuffling instructions, by using the AIE API transpose function ((AMD, 2025b)), such that both data within tiles and the tiles themselves are in column-major order.
Subsequently, we apply similar transformations to matrix BB as demonstrated for matrix AA, across the NPU hierarchy (Fig. 4).

Furthermore, when matrix BB is in row-major, only one 4D transformation is required in the MemTiles (parameters: ss, tt, kctk\_{\text{ct}}, nctn\_{\text{ct}}), since each MemTile holds a kctk\_{\text{ct}} ×\times nctn\_{\text{ct}} tile (Sec. 4.2).
In a similar fashion, matrix CC tiles (row-major) entail a single 4D transformation in the MemTiles (parameters: rr, tt, mctm\_{\text{ct}}, nctn\_{\text{ct}}).

![Figure 5. Simplified view of outer-most (fourth) GEMM tiling level, determined by NPU–DRAM transfers (mrowsm_{\text{rows}}, ncols=4n_{\text{cols}}=4).](https://arxiv.org/html/2512.13282/2512.13282v1/x6.png)

### 4.4. Data Movement Between NPU & DRAM

The outer-most (fourth) level of GEMM tiling encompasses data movement between the NPU and DRAM.
The NPU interfaces with DRAM via ShimTiles and the control of data movement is orchestrated by the on-chip command processor.
ShimTiles are equipped with an input task queue to facilitate DMA transfers, where tasks are submitted sequentially.
When a task terminates, it issues a task-completion token, which the command processor uses to synchronize between multiple DRAM transfers.
Each ShimTile has access to 16 BDs ((AMD, 2025t)), which are used to specify DMA transfers.
When a complex data movement pattern requires more than 16 BDs on a ShimTile, we can reuse (reconfigure) BDs.
Before reconfiguring a BD, it is necessary to properly synchronize and ensure that the previous transfer associated with the BD has completed ((AMD, 2025q)).

Fig. 5 illustrates a simplified view of the data movement between the NPU and DRAM.
Each BD is utilized to describe data movements in a fine-grained manner.
In particular, for matrix AA, each BD defines one ShimTile DMA transfer of mctm\_{\text{ct}} ×\times KK size.
Similarly, for matrix BB, each BD defines a transfer of KK ×\times nctn\_{\text{ct}}.
For matrix CC, each BD describes a transfer of (mctm\_{\text{ct}} ⋅\cdot mrowsm\_{\text{rows}}) ×\times nctn\_{\text{ct}}, since mrowsm\_{\text{rows}} output tiles are aggregated per column.
The command processor program in our implementation maps and inserts BDs into ShimTile task queues as determined by the GEMM mapping strategy (*e.g.,* B​DA​0BD\_{A0}, B​DB​0BD\_{B0}, and B​DC​0BD\_{C0} to ShimTile 0, B​DA​1BD\_{A1}, B​DB​1BD\_{B1}, and B​DC​2BD\_{C2} to ShimTile 1, etc).
Moreover, B​DA​4BD\_{A4}, B​DB​0BD\_{B0}, and B​DC​1BD\_{C1} are also pushed into the queue of ShimTile 0, as dictated by GEMM tiling.
Similarly, our program enqueues BDs describing the GEMM transfers into the input task queues of each ShimTile, in a sequential fashion.
We note that the simplified example in Fig. 5 is directly applicable to XDNA (ncols=4n\_{\text{cols}}=4, see Sec. 4.2), while the BD mapping for XDNA2 can be determined in straightforward fashion.

Depending on the target GEMM dimensions, (*i.e.,* MM, KK, NN), more than the maximum of 16 BDs in each ShimTile might be required.
BD reconfiguration is needed in this situation, which might result in performance degradation (Sec. 5.3.3).
To address this challenge, we propose the following procedure, which enables DMA data transfers to overlap with BD reconfiguration.
Initially, we submit to the task queue five BDs for each of the three AA, BB, and CC DMA transfers.
This efficiently utilizes 15 out of the 16 BDs available in each ShimTile.
Each DMA transfer begins immediately once its associated BD reaches the front of the queue.
For instance, B​DA​0BD\_{A0} and B​DB​0BD\_{B0} transfers start first in Fig 5.
Subsequently, the command processor waits for a task-completion token for each output transfer in a sequential manner (*e.g.,* first for B​DC​0BD\_{C0}, then for B​DC​1BD\_{C1}, etc).
Notice that the command processor only needs to wait for completion of each BD associated with the output matrix (*e.g.,* B​DC​0BD\_{C0}), since once it completes, the corresponding input BDs (*e.g.,* B​DA​0BD\_{A0} and B​DB​0BD\_{B0}) have also finished.
Therefore, once each output BD completes, the three retired BDs can be safely reconfigured, and the next three BDs are inserted into the queue (if available).
This ensures that 15 BDs are in the queue in the steady-state operation, allowing DMA data movement to overlap efficiently with BD reconfiguration.
This process is repeated iteratively until GEMM tiling is completed.

The fine-grained BD description of NPU–DRAM data movement discussed above allows supporting very large GEMM dimensions.
GEMM dimensionality is limited by the NPU registers bitwidth utilized in multi-dimensional tensor addressing ((AMD, 2025d)).
For instance, when storing BB in column-major, the GEMM programming example in ((AMD, 2025n)) allows only a reduction KK dimension of up to ∼\sim4K for bf16 on XDNA2, while our approach allows sizes >> 64K in all dimensions.

### 4.5. Analytical Modeling Optimization

To maximize GEMM performance, we propose an optimization methodology based on analytical modeling.
First, we focus on single-core performance to gain insights for the on-chip compute part,
and then extend our methodology to optimize the system-level performance by incorporating off-chip DRAM BW constraints.

#### 4.5.1. Single-Core GEMM Optimization

Our model utilizes architectural parameters, such as the peak compute throughput of the cores (p​e​a​k​\_​M​A​C​speak\\_MACs in MACs/cycle) and the DMA BW (D​M​A​\_​B​WDMA\\_BW in Bytes/cycle), in order to identify the optimal mctm\_{\text{ct}}, kctk\_{\text{ct}}, nctn\_{\text{ct}} parameters that maximize performance.
We define the efficiency (e​f​feff) as the fraction of the attained compute throughput to the peak throughput of the core.
Eq. 1 expresses the compute cycles of the GEMM kernel (Cc​o​m​pC\_{comp}), while Eq. 2 and 3 formulate the number of cycles needed for the DMA transfers of AA (C​Ac​o​m​mCA\_{comm}) and BB (C​Bc​o​m​mCB\_{comm}) tiles, respectively.
Here, t​y​(⋅)ty(\cdot) indicates the data type size (in Bytes) of the matrices.

| (1) |  | Cc​o​m​p=mct⋅kct⋅nct/(e​f​f⋅p​e​a​k​\_​M​A​C​s)\displaystyle C\_{comp}=m\_{\text{ct}}\cdot k\_{\text{ct}}\cdot n\_{\text{ct}}/(eff\cdot peak\\_MACs) |  |
| --- | --- | --- | --- |
| (2) |  | C​Ac​o​m​m=mct⋅kct⋅t​y​(A)/D​M​A​\_​B​W\displaystyle CA\_{comm}=m\_{\text{ct}}\cdot k\_{\text{ct}}\cdot ty(A)/DMA\\_BW |  |
| --- | --- | --- | --- |
| (3) |  | C​Bc​o​m​m=kct⋅nct⋅t​y​(B)/D​M​A​\_​B​W\displaystyle CB\_{comm}=k\_{\text{ct}}\cdot n\_{\text{ct}}\cdot ty(B)/DMA\\_BW |  |
| --- | --- | --- | --- |

Next, we define the constraint in Eq. 4 to ensure that the single-core GEMM remains compute bound (not bounded by the DMA BW for AA and BB tiles).
Furthermore, Eq. 5 restricts the GEMM buffers to fit within the 64KB L1 memory capacity (with 1KB reserved for stack).
Finally, we impose the straightforward constraint that mctm\_{\text{ct}}, kctk\_{\text{ct}}, nctn\_{\text{ct}} need to be multiples of rr, ss, tt, respectively (not shown).

| (4) |  | Cc​o​m​p≥{C​Ac​o​m​m,C​Bc​o​m​m}\displaystyle C\_{comp}\geq\{CA\_{comm},\ CB\_{comm}\} |  |
| --- | --- | --- | --- |
| (5) |  | {2⋅mct⋅kct⋅ty(A)+2⋅kct⋅nct⋅t​y​(B)+mct⋅nct⋅ty(C)}≤63KB\displaystyle\begin{split}\{2\cdot m\_{\text{ct}}\cdot k\_{\text{ct}}\cdot ty(A)&+2\cdot k\_{\text{ct}}\cdot n\_{\text{ct}}\cdot ty(B)\\ &+m\_{\text{ct}}\cdot n\_{\text{ct}}\cdot ty(C)\}\leq 63\,\text{KB}\end{split} |  |
| --- | --- | --- | --- |

The solution of mctm\_{\text{ct}}, kctk\_{\text{ct}}, nctn\_{\text{ct}} can be formulated as an integer programming (IP) optimization problem utilizing the aforementioned constraints.
The IP is solved exhaustively by setting the maximization of the number of MACs (mctm\_{\text{ct}} ⋅\cdot kctk\_{\text{ct}} ⋅\cdot nctn\_{\text{ct}}) as the main objective.
This increases data reuse in GEMM, thereby maximizing the overall efficiency.
Furthermore, due to the output stationary GEMM mapping, we impose a second objective to minimize the output CC tile (*i.e.,* the product mctm\_{\text{ct}} ⋅\cdot nctn\_{\text{ct}}).
This is essential in reducing the number of loads/stores for accumulations
and decreasing memory stalls caused by bank conflicts.
Evidently, the two optimization objectives lead to increased kctk\_{\text{ct}} and reduced mctm\_{\text{ct}}, nctn\_{\text{ct}} (sufficiently large to not become DMA BW bound).
Moreover, notice that we do not impose DMA constraints on the output CC buffer, as opposed to AA and BB (Eq. 2, 3), while also retaining CC as a single buffer (Eq. 5).
This significantly increases the search space, thus increasing performance in the general GEMM case (Sec. 5.3.2), which is an essential aspect of the system-level optimization discussed below.

#### 4.5.2. System-Level NPU Array Optimization

First, we analytically express the DRAM accesses for each matrix in GEMM.
Eq. 6 captures the DRAM reads needed for matrix AA (Am​e​mA\_{mem}).
The first term, mct⋅mrows⋅K⋅t​y​(A)m\_{\text{ct}}\cdot m\_{\text{rows}}\cdot K\cdot ty(A), represents the DRAM reads (in Bytes) during GEMM tiling along the KK dimension.
This is due to the output stationary mapping and because AA is broadcast across rows (Sec. 4.2).
The second term, N/(nct⋅ncols)N/(n\_{\text{ct}}\cdot n\_{\text{cols}}), describes the repeat factor of the aforementioned read accesses due to tiling along the NN dimension.
In a similar manner, the third term, M/(mct⋅mrows)M/(m\_{\text{ct}}\cdot m\_{\text{rows}}), captures the repeat across the MM dimension.

|  | Am​e​m=(mct⋅mrows⋅K⋅t​y​(A))​(Nnct⋅ncols)​(Mmct⋅mrows)\displaystyle A\_{mem}=\left(m\_{\text{ct}}\cdot m\_{\text{rows}}\cdot K\cdot ty(A)\right)\left(\frac{N}{n\_{\text{ct}}\cdot n\_{\text{cols}}}\right)\left(\frac{M}{m\_{\text{ct}}\cdot m\_{\text{rows}}}\right) |  |
| --- | --- | --- |
| (6) |  | ⇒Am​e​m=M⋅K⋅N⋅t​y​(A)/(nct⋅ncols)\displaystyle\Rightarrow\ A\_{mem}=M\cdot K\cdot N\cdot ty(A)/(n\_{\text{ct}}\cdot n\_{\text{cols}}) |  |
| --- | --- | --- | --- |

Similarly, Eq. 7 represents the DRAM reads for matrix BB (Bm​e​mB\_{mem}), while Eq. 8 shows the DRAM writes for the output matrix CC (Cm​e​mC\_{mem}).
We note that these equations provide an elegant and compact representation of data reuse and tiling scheme in GEMM.

| (7) |  | Bm​e​m=M⋅K⋅N⋅t​y​(B)/(mct⋅mrows)\displaystyle B\_{mem}=M\cdot K\cdot N\cdot ty(B)/(m\_{\text{ct}}\cdot m\_{\text{rows}}) |  |
| --- | --- | --- | --- |
| (8) |  | Cm​e​m=M⋅N⋅t​y​(C)\displaystyle C\_{mem}=M\cdot N\cdot ty(C) |  |
| --- | --- | --- | --- |

Furthermore, Eq. 9 models the GEMM compute time on the NPU (Tc​o​m​pT\_{comp}), while Eq. 10 expresses the total DRAM access time (Tm​e​mT\_{mem}).
Here, p​e​a​k​\_​T​O​P​Speak\\_TOPS denotes the theoretical peak throughput of the NPU array, calculated at the maximum operating frequency.
Furthermore, D​R​A​M​\_​B​WDRAM\\_BW is the effective DRAM BW achieved during GEMM execution on the NPU.
Note that, since all NPU cores execute the same GEMM kernel independently, the single-core efficiency e​f​feff, as defined in Sec. 4.5.1, directly corresponds to the entire NPU array efficiency in Eq. 9.

| (9) |  | Tc​o​m​p=2⋅M⋅K⋅N/(e​f​f⋅p​e​a​k​\_​T​O​P​S)\displaystyle T\_{comp}=2\cdot M\cdot K\cdot N/(eff\cdot peak\\_TOPS) |  |
| --- | --- | --- | --- |
| (10) |  | Tm​e​m=(Am​e​m+Bm​e​m+Cm​e​m)/D​R​A​M​\_​B​W\displaystyle T\_{mem}=(A\_{mem}+B\_{mem}+C\_{mem})/DRAM\\_BW |  |
| --- | --- | --- | --- |

As discussed previously (Sec. 4.5.1), minimizing mctm\_{\text{ct}} and nctn\_{\text{ct}}, as well as maximizing kctk\_{\text{ct}} is essential to attain high efficiency, and thus maximized GEMM performance (equivalent to minimized Tc​o​m​pT\_{comp}).
However, from Eq. 6 and 7, we notice that nctn\_{\text{ct}} and mctm\_{\text{ct}} parameters appear on the denominator of DRAM accesses for AA and BB, respectively.
Therefore, the DRAM access time, Tm​e​mT\_{mem}, increases as mctm\_{\text{ct}} and nctn\_{\text{ct}} decrease.
This highlights the inverse relationship between compute and memory time: as one increases, the other decreases.
Thus, the optimal GEMM performance is attained at the balanced point where they intersect (*i.e.,* Tc​o​m​pT\_{comp} ≈\approx Tm​e​mT\_{mem}).

In order to find the aforementioned optimal balanced point we empirically set starting values for mctm\_{\text{ct}}, kctk\_{\text{ct}}, nctn\_{\text{ct}}, and e​f​feff parameters, and perform the iterative procedure explained below (starting values reduce iterations to typically <<5).
These starting values are set based on the results of the single-core kernel optimization (Sec. 4.5.1), and the effective D​R​A​M​\_​B​WDRAM\\_BW during GEMM execution (measured via micro-benchmarking, refer to Sec. 5.2.1).
First, we measure the actual GEMM performance on the NPU device as well as the single kernel efficiency for the starting values, verifying that GEMM is memory bound (Tc​o​m​pT\_{comp} << Tm​e​mT\_{mem}, due to low mctm\_{\text{ct}}, nctn\_{\text{ct}}, and high kctk\_{\text{ct}}).
In each iteration, we decrease the parameter kctk\_{\text{ct}} (as a multiple of ss), and solve exhaustively an IP similar to the previous Sec. (4.5.1).
However, for that specific iteration, we fix the kctk\_{\text{ct}} parameter and the objective is to maximize the product of mctm\_{\text{ct}} ⋅\cdot nctn\_{\text{ct}}.
This leads to maximized mctm\_{\text{ct}}, nctn\_{\text{ct}} values, given the DMA BW and L1 memory constraints, ensuring a highly optimized GEMM kernel (maximized number of MACs).
This is essential in identifying the optimal balanced point, because it results in the smallest possible increment of Tc​o​m​pT\_{comp} in each iteration, while also ensuring maximized possible GEMM performance given the specific parameters of that iteration.
We note here that e​f​feff is estimated based on each previous point measurements of each iteration.
We iteratively measure GEMM performance on the NPU device of the top-ranked solution for each IP, verifying that in each step performance is higher compared to the previous.
While GEMM performance is increasing in each step, at some point we observe lower performance, and the iteration stops.
Evidently, at this specific point GEMM has become compute bound (Tc​o​m​pT\_{comp} >> Tm​e​mT\_{mem}), while also performance is lower.
Therefore, the balanced point where GEMM performance is maximized has been identified (mctm\_{\text{ct}}, kctk\_{\text{ct}}, nctn\_{\text{ct}} parameters of the previous iteration).

## 5. Evaluation

For the experimental evaluation, we use two representative mini PCs, corresponding to two NPU generations.
The Minisforum UM790 Pro ((Minisforum, 2025)), equipped with Ryzen 9 7940HS processor (Phoenix Point) ((AMD, 2025h)), is used for XDNA.
For XDNA2, we use the ASRock 4×\times4 Box ((Industrial, 2025)), featuring the AMD Ryzen AI 7 350 processor (Krackan Point) ((AMD, 2025i)).
Both mini PCs have dual-channel DDR5-5600 MT/s DRAM.
The CPUs run Ubuntu 24.04 LTS and serve as the host.
In particular, CPUs allocate
buffers in DRAM using Xilinx Runtime (XRT) ((AMD, 2025v)), configure the NPU, invoke the NPU for GEMM execution, and process the completion notification.
Finally, throughout all experiments, NPUs are configured at their maximum performance level (*i.e.,* turbo mode ((AMD, 2025o)), utilizing XDNA driver commands ((AMD, 2025j))).

Table 1. Single-core GEMM results for XDNA & XDNA2.

| Dev. | Precision | Kernel Size (In-Out, m_ct × k_ct × n_ct) | Throughput (MACs/cycle) | L1 Core Mem. (KB) |
| --- | --- | --- | --- | --- |
| XDNA | int8-int8 | 64 × 232 × 64 | 233.0 | 62.0 (97%) |
| XDNA | int8-int16 | 64 × 216 × 64 | 217.6 | 62.0 (97%) |
| XDNA | int8-int32 | 48 × 280 × 48 | 192.0 | 61.5 (96%) |
| XDNA | bf16-bf16 | 64 × 104 × 64 | 112.6 | 60.0 (94%) |
| XDNA2 | int8-int8 | 64 × 232 × 64 | 450.6 | 62.0 (97%) |
| XDNA2 | int8-int16 | 64 × 216 × 64 | 419.8 | 62.0 (97%) |
| XDNA2 | int8-int32 | 48 × 280 × 48 | 384.0 | 61.5 (96%) |
| XDNA2 | bf16-bf16 | 48 × 152 × 48 | 158.1 | 61.5 (96%) |

### 5.1. Single-Core GEMM Performance

We exploit the AIE API ((AMD, 2025a)) to design highly optimized single-core GEMM kernels.
The kernels are compiled using the *xchesscc* tool, employing various compiler directives (*e.g.,* software pipelining and loop unrolling/flattening) to attain high efficiency.
Performance is assessed via hardware profiling utilizing the NPU trace unit ((AMD, 2025t, s)), thereby enabling cycle accurate measurements.
For int8, besides full output precision (int32), we also perform precision reduction to 16- and 8-bits, which is a common technique to increase GEMM performance in AIE architectures ((AMD, 2025u; Zhuang et al., 2024; Deng et al., 2024; Mhatre et al., 2025)).
Moreover, for bf16, we retain the output precision to bf16.
Finally, for XDNA2 we emulate the bf16 precision utilizing the bfp16 hardware datapath, which leads to increased GEMM performance (emulation attained via a specific *xchesscc* flag at compile time, as mentioned in ((AMD, 2025a))).

In Table 1, we present the top-ranked solutions of the single-core optimization procedure described in Sec. 4.5.1.
For int8 precision, we attain very high throughput, ranging from 192.0–233.0 MACs/cycle and from 384.0–450.6 MACs/cycle for XDNA and XDNA2, respectively.
Similarly for bf16, we achieve 112.6 MACs/cycle for XDNA, and 158.1 MACs/cycle for XDNA2.
Observe that in both cases, XDNA2 attains higher throughput compared to XDNA, since it has higher peak compute throughput capabilities per core.
We note that, since performance is measured via hardware tracing, the results in Table 1 include the inevitable memory stalls due to bank conflicts, thus reflecting the actual NPU performance (when not bounded by
DRAM BW).
Finally, very high L1 memory usage is achieved across all solutions, ranging from 94–97%.

Table 2. Evaluation of two top-ranked solutions for XDNA across various data types (BB column-major).

| Precision | Kernel Size (In-Out, m_ct × k_ct × n_ct) | Product (m_ct · n_ct) | Thrghpt. (MACs/cyc) | L1 Core Mem. (KB) | L2 Total Mem. (KB) | Peak Comp. (TOPS) | GEMM Size (MM × KK × NN) | Actual (NPU TOPS) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| int8-int8 | 112 × 112 × 112 | 12.3K | 212.5 | 61.3 (96%) | 980 (48%) | 6.80 | 4032 × 4032 × 4032 | 6.52 |
| int8-int8 | 112 × 104 × 128 | 14.0K | 207.4 | 62.8 (98%) | 1004 (49%) | 6.63 | 4032 × 4160 × 4096 | 6.48 |
| int8-int16 | 96 × 112 × 96 | 9.0K | 192.0 | 60.0 (94%) | 960 (47%) | 6.14 | 4224 × 4032 × 4224 | 5.85 |
| int8-int16 | 80 × 104 × 128 | 10.0K | 186.9 | 62.3 (97%) | 996 (49%) | 5.98 | 4160 × 4160 × 4096 | 5.75 |
| int8-int32 | 80 × 88 × 96 | 7.5K | 146.0 | 60.3 (94%) | 964 (47%) | 4.67 | 4160 × 4224 × 4224 | 4.42 |
| int8-int32 | 64 × 80 × 128 | 8.0K | 133.1 | 62.0 (97%) | 992 (48%) | 4.26 | 4096 × 4160 × 4096 | 4.09 |
| bf16-bf16 | 96 × 56 × 96 | 9.0K | 99.8 | 60.0 (94%) | 960 (47%) | 3.19 | 4224 × 4032 × 4224 | 3.12 |
| bf16-bf16 | 96 × 48 × 112 | 10.5K | 97.3 | 60.0 (94%) | 960 (47%) | 3.11 | 4224 × 4032 × 4032 | 3.02 |

Table 3. Evaluation of two top-ranked solutions for XDNA2 across various data types (BB column-major).

| Precision | Kernel Size (In-Out, m_ct × k_ct × n_ct) | Product (m_ct · n_ct) | Thrghpt. (MACs/cyc) | L1 Core Mem. (KB) | L2 Total Mem. (KB) | Peak Comp. (TOPS) | GEMM Size (MM × KK × NN) | Actual (NPU TOPS) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| int8-int8 | 144 × 72 × 144 | 20.3K | 343.0 | 60.8 (95%) | 2106 (51%) | 39.52 | 4032 × 4320 × 4608 | 37.35 |
| int8-int8 | 160 × 64 × 144 | 22.5K | 322.6 | 60.5 (95%) | 2064 (50%) | 37.16 | 4480 × 4224 × 4608 | 36.13 |
| int8-int16 | 128 × 72 × 112 | 14.0K | 307.2 | 61.8 (97%) | 2084 (51%) | 35.39 | 4096 × 4320 × 4480 | 30.77 |
| int8-int16 | 160 × 64 × 96 | 15.0K | 271.4 | 62.0 (97%) | 2016 (49%) | 31.26 | 4480 × 4224 × 4608 | 29.59 |
| int8-int32 | 96 × 64 × 96 | 9.0K | 256.0 | 60.0 (94%) | 2016 (49%) | 29.49 | 4224 × 4224 × 4608 | 24.74 |
| int8-int32 | 128 × 56 × 80 | 10.0K | 209.9 | 62.3 (97%) | 2036 (50%) | 24.18 | 4096 × 4032 × 4480 | 21.67 |
| bf16-bf16 | 112 × 48 × 96 | 10.5K | 137.2 | 60.0 (94%) | 2496 (61%) | 15.81 | 4032 × 4224 × 4608 | 14.52 |
| bf16-bf16 | 160 × 40 × 80 | 12.5K | 124.1 | 62.5 (98%) | 2400 (59%) | 14.30 | 4480 × 4160 × 4480 | 13.67 |

### 5.2. GEMM Performance on NPU Array

In this section, we present GEMM performance on the entire NPU array.
Performance is evaluated using wall-clock time, thereby capturing the actual performance observed by the users (includes OS overheads, NPU dispatch time, etc.) ((AMD, 2025r)).
All reported results represent the average of 100 runs.

#### 5.2.1. Optimal Balanced GEMM Kernel

As shown in the previous section, GEMM kernels can attain very high throughput.
However, when using the optimum compute kernel sizes of Table 1, we observe that GEMM performance on the NPU array remains low.
For instance, for int8-int16 precision, we obtain only 17.86 TOPS on XDNA2 at ∼\sim4K square GEMM size.
However, the peak compute capability of this kernel on the XDNA2 array is 48.36 TOPS, when calculated at maximum operating frequency (identified through XDNA driver commands ((AMD, 2025j)), *i.e.,* 1 GHz and 1.8 GHz for XDNA and XDNA2, respectively).
This implies that GEMM is memory bound at this specific kernel size, due to low values of mctm\_{\text{ct}} and nctn\_{\text{ct}} (inverse relationship of compute and memory).
Therefore, we leverage the optimization methodology described in Sec. 4.5.2 in order to identify the optimal balanced kernel (*i.e.,* where compute and memory become balanced).
First, we use micro-benchmarking to estimate the effective DRAM BW that is available to the NPU when running GEMM workloads.
For micro-benchmarking, we imitate GEMM transfers from DRAM to NPU array and vice-versa, observing ∼\sim15 GB/s and ∼\sim50 GB/s for XDNA and XDNA2, respectively.
Afterwards, we exploit these values to select starting points for the iterative optimization procedure of Sec. 4.5.2.

Tables 2 and 3 present the two top-ranked solutions, for XDNA and XDNA2, respectively.
First, notice that kernel sizes have lower kctk\_{\text{ct}} and higher mctm\_{\text{ct}}, nctn\_{\text{ct}} compared to kernels in Table 1.
This decreases their compute throughput; for example, 96×\times112×\times96 for int8-int16 (Table 2) achieves 192.0 MACs/cycle, compared to 217.6 MACs/cycle of 64×\times216×\times64 (Table 1).
However, GEMM performance on the NPU array increases, since compute and memory become balanced.
For example, when using the 96×\times112×\times96 kernel, GEMM performance on XDNA for ∼\sim4K GEMM size is 5.85 TOPS (Table 2) compared to 4.03 TOPS of 64×\times216×\times64 (not shown in Table 1).

Second, observe that when further increasing the product of mct⋅nctm\_{\text{ct}}\cdot n\_{\text{ct}} (thus decreasing kctk\_{\text{ct}} to fit in L1), GEMM performance on NPU array drops.
For instance, for int8-int8, kernel 160×\times64×\times144 has higher mct⋅nctm\_{\text{ct}}\cdot n\_{\text{ct}} product compared to 144×\times72×\times144 (22.5K *vs.* 20.3K), but performance is lower: 36.13 *vs.* 37.35 TOPS (Table 3).
This is because at this point GEMM has become compute bound, but compute has also become lower (throughput drops to 322.6 from 343.0 MACs/cycle).
The peak compute throughput on the entire XDNA2 array when attaining single-core throughput of 322.6 MACs/cycle is 37.16 TOPS.
However, the actual GEMM performance is 36.13 TOPS for ∼\sim4K GEMM size, as shown in Table 3.
This ∼\sim3% difference is mainly attributed to DMA transfers of CC tiles (single output buffer), occurring at the end of each complete reduction across KK (every KK/kctk\_{\text{ct}} tiles).
Another smaller contribution comes from the vectorized zeroing kernel executing every KK/kctk\_{\text{ct}} tiles (Sec. 4.2), which is typically <<10% of GEMM kernel time (thus, <<0.15% for the ∼\sim4K GEMM of this particular example).
Both output DMA transfers and zeroing kernel cycles have been verified using NPU tracing.
Note that as KK becomes higher, these effects are further amortized.
Other contributions include the DRAM transfer time for initial AA and BB tiles, along with final output CC tiles, as well as overheads due to wall-clock time measurements (*e.g.,* NPU dispatch time) ((AMD, 2025r)).
In summary, the bolded solutions across all precisions in Tables 2 and 3 represent the optimal balanced point between compute and memory, where GEMM performance is maximized.

We note that results in Tables 2, 3 correspond to matrix BB in column-major (results for BB in row-major are presented in Sec. 5.2.3).
Across all results, the contiguous parameter kmtk\_{\text{mt}}, which is utilized to increase the effective DRAM BW (and hence GEMM performance), is configured as shown below (Sec. 5.2.2).
Finally, the optimization procedure requires less than 30 minutes to identify the optimum solution for each data type (evaluated on the corresponding mini PCs).
The overall execution time is dominated by the compilation time of *xchesscc* and IRON (up to 5 minutes per iteration), while the exhaustive search takes less than 1 s in all cases.

#### 5.2.2. Contiguous kmtk\_{\text{mt}} Parameter

In this section, we present the impact of the contiguous kmtk\_{\text{mt}} parameter on GEMM performance.
Fig. 6 depicts the GEMM performance when varying the parameter kmtk\_{\text{mt}} and utilizing the optimal balanced kernels for two arbitrary data types (∼\sim4K GEMM size and BB in column-major).
For instance, for XDNA, when kmtk\_{\text{mt}} is equal to kctk\_{\text{ct}} (== 56 in Fig. 6a), GEMM performance is very low (*i.e.,* 1.27 TOPS).
As kmtk\_{\text{mt}} increases (in multiples of kctk\_{\text{ct}}) performance improves.
This is because it enables higher effective DRAM BW, by traversing more contiguous elements for both AA (row-major) and BB (column-major) matrices.
However, at some point, this enhancement becomes saturated (*i.e.,* higher kmtk\_{\text{mt}} provides marginal improvement).
Therefore, we empirically select the smaller value where GEMM performance becomes saturated for each data type (*i.e.,* kmtk\_{\text{mt}}=224 in Fig. 6a).
Note that selecting the smallest value while also maintaining high GEMM performance is essential, since it reduces any potential zero padding needed to align the final GEMM dimensions with the native GEMM size (Sec. 4.2).
For example, for the bf16-bf16 case, the native GEMM size operating natively on the entire 4×\times4 XDNA array is 384×\times224×\times384.
We note here that this results in reduced L2 memory utilization, ranging from 47–61% across all data types (Tables 2 & 3).
While maximized L2 memory usage can be attained for higher kmtk\_{\text{mt}} values (*e.g.,* 96% for kmtk\_{\text{mt}}=560 in Fig. 6a), this only leads to marginal performance improvement in GEMM (<< 1%).

Similarly, for XDNA2, for int8-int16, we set kmtk\_{\text{mt}}=432 (Fig. 6b).
In this case, the native GEMM size on the XDNA2 array becomes 512×\times432×\times896.
We note here that the three latest points of Fig. 6b are enabled via the neighboring memory sharing in MemTiles, as a direct result of the GEMM mapping strategy on XDNA2 (Sec. 4.2.2).
For all other data types, we set the parameter kmtk\_{\text{mt}} in a similar fashion.
Specifically, for XDNA, we set kmtk\_{\text{mt}} equal to 448 for int8-int8 and int8-int16, while for int8-int32 we set it to 352.
On XDNA2, we use 432 for int8-int8, and 384 for both int8-int32 and bf16-bf16.

It is important to mention here that the non-optimized GEMM example in ((AMD, 2025n)), cannot support sufficient contiguous elements.
For a fair comparison, we utilize our optimized balanced kernels and modify their implementation to support single output CC buffers (allowing it to fit in L1).
For instance, for the data types shown in Fig. 6, we attain 2.4×\times and 3.6×\times higher performance for XDNA and XDNA2, respectively.
These results highlight the importance of accessing sufficient contiguous elements in GEMM performance.

![(a) XDNA](https://arxiv.org/html/2512.13282/2512.13282v1/x7.png)

#### 5.2.3. GEMM Performance Sweeps

In Fig. 7 and 8, we show roofline GEMM performance sweeps (in linear scale).
Each point represents a matrix size that is a multiple of the native GEMM size (using the optimal balanced kernel).
We select more than 400 points for each case (separately for BB in column- and row-major), up to 8K-sized matrices, without favoring any particular MM, KK, NN dimension.
First, notice that when arithmetic intensity (ARI) is low (*i.e.,* small matrix sizes), performance is limited.
In this case, GEMM is severely memory bound.
However, as ARI increases, GEMM performance improves, and becomes more stabilized after a specific value.
In all cases, storing matrix BB in column-major provides, on average, higher performance compared to row-major.
This is because of accessing sufficient contiguous data for both AA and BB matrices (determined by kmtk\_{\text{mt}} parameter).
However, for BB in row-major, the contiguous access is limited to the nctn\_{\text{ct}} parameter, while only for AA (row-major), kmtk\_{\text{mt}} contiguous data are traversed.
To this end, for XDNA, we observe, on average, 4.8%, 4.4%, and 0.57% higher performance, for int8-int8, int8-int16, and bf16-bf16, respectively.

Moreover, we notice that for XDNA2 the difference between column- and row-major is higher.
In particular, we observe, on average, 19.1%, 25.2%, and 8.7% higher performance, for int8-int8, int8-int16, and bf16-bf16, respectively.
This difference between XDNA and XDNA2 is presumably attributed to complex interaction between the NPU NoC, the SoC-level fabric and DRAM ((Rico et al., 2024; Subramon et al., 2023)), which affects the effective DRAM BW that NPU perceives (although both mini PC devices are equipped with the same DRAM).
Also, we note that for both XDNA and XDNA2, the difference between column- and row-major for bf16 is lower compared to int8.
This is because of accessing a larger number of bytes for bf16 across the nctn\_{\text{ct}} dimension (when BB is in row-major), which increases GEMM performance in this case, thereby lowering their difference.

For XDNA, we note that performance gets stabilized after a specific ARI value for BB in both row- and column-major formats (almost resembling a line in Fig. 7).
Moreover, for XDNA2, we observe that for BB in column-major, GEMM performance also resembles a stable line.
However, for BB in row-major it displays a more scattered distribution (Fig. 8).
This is because XDNA2 has higher reliance on the effective DRAM BW (due to attaining significantly higher absolute TOPS values), which is substantially increased and stabilized when accessing sufficient contiguous data across both AA and BB matrices.
For example, for int8-int16 on XDNA2, we measure a variability of only 5% for BB in column-major, while for row-major the variability is 19% (ARI ¿ 1600).
Finally, across all points in GEMM sweeps, XDNA attains up to 6.76 (int8-int8), 6.05 (int8-int16), 4.57 (int8-int32), and 3.14 (bf16-bf16) TOPS.
Similarly, XDNA2 achieves up to 38.05 (int8-int8), 31.52 (int8-int16), 25.31 (int8-int32), and 14.71 (bf16-bf16) TOPS (int8-int32 sweep omitted for brevity).

![(a) int8-int8](https://arxiv.org/html/2512.13282/2512.13282v1/x9.png)

![(a) int8-int8](https://arxiv.org/html/2512.13282/2512.13282v1/x12.png)

### 5.3. Insights & Discussion

#### 5.3.1. Performance Across Multiple GEMM Sizes in DL Workloads

Modern DL workloads perform GEMMs with a wide range of sizes across their layers.
Our employed output stationary mapping allows arbitrary GEMM dimensions to be supported, by applying zero-padding to align with the native GEMM size (Sec. 4.2).
Zero-padding can be applied efficiently by utilizing the NPU’s architectural support for on-the-fly zero-padding in MemTile channels ((AMD, 2025t)); leveraging this feature is left for future work.

Switching between different GEMM sizes can incur critical performance inefficiencies.
To this end, one approach could be to reconfigure the NPU array with a dedicated GEMM design for each size.
When quantifying the reconfiguration latency of the entire GEMM design, we measure a delay of 3.4 ms and 4.9 ms on XDNA and XDNA2, respectively.
However, this reconfiguration latency is comparable to the GEMM execution time (*e.g.,* a ∼\sim4K square GEMM for int8-int16 on XDNA2 takes 5.2 ms).
This underscores that reconfiguring the entire design can impose substantial overheads.

When retaining the same GEMM design on the NPU, only two parameters require reconfiguration across different problem sizes (M,K,NM,K,N): (i) the total number of output tiles M⋅N/(mct⋅nct)M\cdot N/(m\_{\text{ct}}\cdot n\_{\text{ct}}), and (ii) the number of tiles across the reduction dimension K/kctK/k\_{\text{ct}} ((Rösti and Franz, 2025)).
According to our measurements, this negligible reconfiguration does not incur any noticeable performance overhead at the system level.
Hence, identifying the optimal parameters (*i.e.,* mctm\_{\text{ct}}, kctk\_{\text{ct}}, nctn\_{\text{ct}}, kmtk\_{\text{mt}}, Sec. 4), and reusing them across different GEMM sizes is essential for high-performance DL deployment.
Finally, we note that the system-level GEMM results presented are specific to the mini PCs used in this evaluation.
However, our optimization methodology is generalizable to any NPU device in the current two generations.

#### 5.3.2. Single Output Buffer

Due to output stationary mapping, we retain the output CC tiles as single buffers and apply double-buffering only for inputs AA and BB (Sec. 4.2).
This is crucial in maximizing performance in the general GEMM case, because it enables significantly higher flexibility in single-core parameter optimization.
In particular, it allows increased values for mctm\_{\text{ct}} ×\times kctk\_{\text{ct}} ×\times nctn\_{\text{ct}} parameters, compared to utilizing double-buffering (constrained by L1 memory size).
This results in identifying a balanced kernel that has higher performance, thereby enabling higher GEMM performance at the system level.
For example, when using the optimization methodology described in Sec. 4.5.2 and apply double-buffering for CC, we identify the 112×\times48×\times96 size as the optimal balanced kernel for int8-int16 data type on XDNA2.
For ∼\sim4K square GEMM, this kernel provides 26.1 TOPS.
However, the 128×\times72×\times112 kernel of Table 3 (single CC buffer), provides 30.77 TOPS, representing an 18% performance improvement.
Similarly, on XDNA, double buffer on CC provides 2.76 TOPS (80×\times40×\times96 kernel), while single buffer offers 3.12 TOPS (96×\times56×\times96 kernel on Table 2, 13% higher).
The transfer of the output CC tiles in the single buffer case gets amortized as the reduction KK dimension becomes sufficiently high (typically <<5% degradation in GEMM performance when KK/kctk\_{\text{ct}}>>20).

#### 5.3.3. NPU–DRAM Data Movement & BD Reconfiguration

The procedure delineated in Sec. 4.4 enables efficient overlapping of NPU–DRAM data movement with BD reconfiguration.
To quantify its impact on performance, we modify our design to synchronize and reconfigure BDs sequentially (without overlap).
In this case, for int8-int16 on XDNA2, we notice only 22.21 TOPS at ∼\sim4K square GEMM, while the overlapped design of Table 3 exhibits 30.77 TOPs (28% decrease for non-overlapped design).
Similarly, for XDNA, for int8-int16 (Table 2), we observe a 27% degradation in performance.
This highlights the critical role of overlapping DMA transfers with BD reconfiguration in attaining high GEMM performance.

#### 5.3.4. Future Research

XDNA2 incorporates hardware support for bfp16 precision, where multiple numbers share one common exponent.
This incurs additional challenges for data layout transformations exploiting the multi-dimensional DMA addressing features of the NPUs (Sec. 4.3).
However, this is beyond the scope of this paper and will therefore be addressed in future work.
Furthermore, our proposed optimization methodology can be also be exploited for special cases of GEMM such as general matrix-vector multiplication (GEMV), which we also leave as future work.

## 6. Conclusion

In this work, we propose a novel optimization methodology to maximize GEMM performance on Ryzen AI NPUs.
We observe the inverse relationship between compute and off-chip memory and determine the optimal balanced point, where performance is maximized.
To identify this optimal performance point, we exploit analytical modeling and hardware profiling techniques.
Our methodology attains state-of-the-art GEMM performance and is generalizable across the current AMD Ryzen AI NPU generations.

