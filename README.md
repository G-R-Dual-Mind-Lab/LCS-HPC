# LCS-HPC

## Overview

This project implements the Longest Common Subsequence (LCS) algorithm, a classic dynamic programming problem used to find the longest subsequence common to two sequences. LCS is widely used in bioinformatics, text comparison, and version control systems.

## Implemented Solutions

This repository provides three different implementations of the LCS algorithm:

1. **Sequential Solution with Space Optimization**  
   A standard dynamic programming approach executed on a single core, which keeps only two rows of the DP matrix in memory during execution to reduce space complexity.

2. **Multi-Threading Parallel Solution**  
   Utilizes PThreads and the Worksharing design pattern to accelerate computation by dividing the workload among threads.  
   This solution adopts a wavefront approach, allowing computation to proceed along antidiagonals of the DP matrix. For each antidiagonal, each thread processes a subset of its elements.  
   Two variants are provided: one that stores the entire matrix in memory, and a space-optimized version that maintains only three diagonals at any time.

   ![PthreadsSol1](https://github.com/user-attachments/assets/fa035a46-5dac-4e48-a437-5ac0e7d4e5ff)
   ![PthreadsSol2](https://github.com/user-attachments/assets/94ff4fd9-fc58-4b2c-a9ef-4c448454fed1)
   

4. **Distributed Solution**  
   Implements a hybrid approach using MPI, PThreads, and OpenMP to process large sequences efficiently across multiple machines or nodes.  
   This solution follows a Manager-Worker paradigm (also known as Distributed Bag of Tasks or Processors Farm) and employs a two-level tiling strategy:
   - The master MPI process divides the DP matrix into tiles and creates three threads using PThreads:
     * **Producer thread:** Generates tasks, each corresponding to a tile.
     * **Sender thread:** Sends tasks to idle workers, receives results, and injects computed values into the dependencies of tasks to be sent.
     * **Auxiliary sender thread:** Handles rare cases where the sender tries to inject dependencies into tasks not yet produced by the producer. The auxiliary sender waits until these tasks are ready to be sent.
   - Each worker MPI process works on one tile at a time. Upon receiving a tile, the worker:
     * Performs a second level of tiling, dividing each tile into sub-tiles.
     * Uses OpenMP and the Worksharing design pattern to compute the elements of the received tile.
     * Specifically, the worker applies a wavefront (antidiagonal) approach at the sub-tile level: it processes antidiagonals of sub-tiles, and for each antidiagonal, each thread works on a subset of sub-tiles in that diagonal.

   ![MPISol1](https://github.com/user-attachments/assets/e617f0d2-0ad2-485a-95eb-ac72a7e2732e)
   ![MPISol2](https://github.com/user-attachments/assets/d1340312-eac2-45b7-98fe-2a9fe0ee838c)
   ![MPISol3](https://github.com/user-attachments/assets/63d142fa-9dc4-45ad-92d3-5dbdd4280b92)
   ![MPISol4](https://github.com/user-attachments/assets/f1ba3851-e193-4ba3-a810-0b04c1a6ddc5)
   ![MPISol5](https://github.com/user-attachments/assets/7260ba36-d76a-463d-b5e0-f877afe3815d)
   ![MPISol6](https://github.com/user-attachments/assets/e9ddea27-0100-4c75-a8e1-4365b0bf0bdf)
   ![MPISol7](https://github.com/user-attachments/assets/6fca57b8-97cd-442e-8413-9603512ba50b)
   

## Presentation

For a detailed explanation of the implementation, design choices, execution times, and performance analysis, see the public Canva presentation:  
[Canva Presentation Link]([https://www.canva.com/your-presentation-link](https://www.canva.com/design/DAGjrn1ybhU/EdsVYw9A2izzAkTIfvtmFg/view?utm_content=DAGjrn1ybhU&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=hf0942c3e2a)
