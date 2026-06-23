---
layout: post
title: "The Factorio POV of GPU Bottlenecks"
date: 2026-05-25 15:00:00 +0200
description: "This weekend I had 3 days off, so I decided to play Factorio. While building my factory and killing biters, I listened to the Dwarkesh Patel podcast with Reiner Pope about serving LLMs. At some point, it hit me that a GPU is basically a Factorio megabase."
permalink: /posts/factorio-gpu.html
katex: true
tags: [gpu, systems, llm]
---

This weekend I had 3 days off, so I decided to play Factorio. While building my factory and killing biters, I listened to the [Dwarkesh Patel podcast with Reiner Pope](https://www.dwarkesh.com/p/reiner-pope) about serving LLMs. At some point, it hit me that a GPU is basically a Factorio megabase.

Not metaphorically in the vague “everything is connected” sense. I mean quite concretely: compute units are assemblers, memory bandwidth is belts, caches are chests, batching is train scheduling, and many performance bugs are just throughput bottlenecks wearing a CUDA hoodie.

Because learning is easier when you can draw lines between concepts, I decided to formalize the analogy. If you have played Factorio before and work, or plan to work, in AI/ML, you might find this post helpful. And if not,  you should take 2 weeks off and play factorio.

## What's Factorio?

For those who do not know what Factorio is, it is a game where you land on an alien planet, mine resources, build machines, automate production, and progressively scale your factory until the factory starts to look less like a factory and more like a distributed systems incident.

Factorio is famous for its capacity to make you time travel. Gift Factorio to an engineer, they will start playing at 8am on a Saturday and the next time they look at the clock, it will be Sunday morning.

The core loop is simple:

1. Mine resources.
2. Move resources with belts, inserters, bots, trains, and pipes.
3. Feed machines.
4. Make products.
5. Discover that the factory is too slow.
6. Find the bottleneck.
7. Repeat forever.

This is also a surprisingly good description of GPU performance engineering.

![Factorio-style diagram mapping VRAM and HBM to memory bandwidth, cache and shared memory, SMs and Tensor Cores, and useful output.](/assets/megabase_gpu_labels.png)

_Figure 1: A GPU as a Factorio megabase. The point is not that every part maps perfectly, but that the bottlenecks rhyme._

## So what's this all about?

Modern GPUs are advertised with absurd compute numbers: TFLOP/s, Tensor Core throughput, FP8 throughput, giant memory bandwidth, and many other numbers that look like they were designed to make your laptop feel insecure.

But if you write a kernel, or serve an LLM, you will often observe something annoying: your code does not get anywhere close to peak compute.

The naive reaction is:

> “Why is my GPU slow? It has so many FLOPs.”

The Factorio equivalent is:

> “Are the assemblers actually busy, or are they waiting for plates?”

That is the whole post.

A GPU is not just a bag of compute. It is a factory. And in a factory, throughput is usually limited by whichever part of the system is least able to keep up: the belts, the inserters, the machines, the train stations, the power grid, the buffers, or the recipes themselves.

## The podcast version: two clocks

In the Dwarkesh Patel conversation, Reiner Pope explains LLM serving with a very simple but powerful mental model. To run a forward pass, there are two clocks ticking:

- How long does the compute take?
- How long does the memory movement take?

The total time is controlled by the slower one:

$$
T = \max(t_{\mathrm{compute}}, t_{\mathrm{memory}})
$$

This is the kind of equation that looks almost too simple to be useful, but it is exactly the kind of equation that becomes powerful when you use it relentlessly. In the podcast, this gets used to reason about batching, KV cache reads, long context, latency, and why serving LLMs is often a memory-bandwidth problem rather than a pure compute problem.

Factorio has the same equation:

```text
T = max(t_crafting, t_logistics)
```

If crafting takes longer than delivery, your assemblers are the bottleneck. If delivery takes longer than crafting, your belts are the bottleneck.

Thus we can say:

```text
T = max(time spent doing math, time spent moving bytes) = max(time spent crafting, time spent feeding the factory)
```

## Peak FLOP/s is the number on the assembler box

Imagine you build a Factorio factory with 10,000 assemblers. You look at the crafting speed of each assembler, and see that each assembler can make 10,000 gears per can and announce:

> “My factory can produce one billion gears per second.”

If you played Factorio, you know this is nonsense. The assemblers only craft if they are fed. You also need enough iron plates, enough belts, enough inserters, enough power, enough output capacity, and enough space to route everything without turning the base into pasta.

GPU peak FLOP/s is similar. It is real, but it is conditional. It tells you how fast the GPU can do math under very favorable conditions. It does not tell you whether your particular program supplies enough data, reuses that data well, accesses memory cleanly, avoids synchronization stalls, and keeps the compute units occupied.

> Peak FLOP/s assumes the assemblers are perfectly fed.

And the first law of megabases is that the assemblers are almost never perfectly fed by accident.

## The bottleneck is not a place, it is a ratio

One of the easiest mistakes to make is to ask:

> “Is this GPU memory-bound or compute-bound?”

as if the GPU itself has a permanent personality.

That is like asking whether a Factorio factory is belt-bound or assembler-bound without specifying the recipe. The same belt might be perfectly fine for one recipe and completely hopeless for another.

The better question is:

> “How much useful work do I get per item delivered?”

This is called operational intensity:

```text
operational intensity = operations / bytes moved
```

If each item delivered from the belt enables lots of crafting, you can keep your assemblers busy. If each delivered item enables only one tiny craft, the belts become the bottleneck.

This is the heart of the Roofline model. The Roofline model says that attainable performance is bounded by the smaller of two quantities: peak compute, and memory bandwidth multiplied by operational intensity. The original paper writes the idea as:

```text
attainable performance = min(peak compute, peak memory bandwidth × operational intensity)
```
![The Roofline model.](/assets/roofline_model.png)

_Figure 3: The Roofline model._

## A cursed recipe: vector add

Take the Iron Gear recipe in Factorio. It consumes two iron plates, runs for less than a second, and produces one output item. If the assembler is extremely fast, the limiting factor is not crafting. It is how quickly the belts can deliver two plates and remove the result.

![iron gear recipe](/assets/iron-gear-recipe.png)

_Iron Gear Recipe_

Vector add is exactly this cursed recipe.

Let us start with one of the simplest GPU kernels, let $a,b \in \mathbb{R}^n$ be some vectors and $c$ their sum:

```math
c[i] = a[i] + b[i]
```

For each element, we load `a[i]`, load `b[i]`, add them, and store `c[i]`.

In FP32, that is roughly:

- 4 bytes loaded for `a[i]`
- 4 bytes loaded for `b[i]`
- 4 bytes stored for `c[i]`
- 1 floating-point addition

So the operational intensity is about:

```text
1 FLOP / 12 bytes ≈ 0.083 FLOP per byte
```

This is terrible.

In Factorio terms, vector add is a recipe where you pull two plates from the main bus, tap them together once, and immediately ship the result away. The assembler barely does anything. Almost all the work is logistics.

To make the numbers concrete, NVIDIA lists the H100 SXM at 67 TFLOP/s of FP32 performance and 3.35 TB/s of memory bandwidth ([H100 Specs](https://www.nvidia.com/fr-fr/data-center/h100/)). A perfect vector add on that GPU would be bounded around:

```text
3.35 TB/s × 0.083 FLOP/byte ≈ 0.28 TFLOP/s
```

That is not close to 67 TFLOP/s. It is not close because vector add does not give the GPU enough math per byte moved. The assemblers are not weak. The recipe is logistics-dominated.

If your mental model is “my GPU has 67 TFLOP/s, so every program should run near 67 TFLOP/s,” this feels disappointing. If your mental model is Factorio, it is obvious. You built a god factory and fed it with one sad belt.

## A beautiful recipe: matrix multiplication

Now compare vector add with matrix multiplication.

Matrix multiplication is the green circuits of machine learning. It is everywhere, it scales beautifully, and if you build the subfactory properly, the same ingredients get reused many times.

In a matrix multiply, a tile of matrix A and a tile of matrix B can be loaded into fast local memory, then reused for many multiply-adds before the GPU needs to go back to slow global memory.

Factorio translation:

> Instead of fetching one iron plate from the main bus, crafting once, and sending the result away, you bring a whole chest of plates next to a dense block of assemblers. Then you craft from that chest over and over before refilling it.

This is why matrix multiplication can reach high hardware utilization. It has high operational intensity. Each byte delivered can participate in many operations.

This is also why techniques like tiling and shared memory matter. The [CUDA Best Practices guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) discusses using shared memory in matrix multiplication to read data from global memory once, reuse it locally, and avoid redundant global memory transfers.

![Matrix multiplication subfactory in Factorio](/assets/factorio-matmul.png)

_Figure 5: Matrix multiplication works well because the factory gets many crafts out of each delivery._

## Coalescing: clean belts beat spaghetti

Memory bandwidth is not just about how many bytes you move. It is also about how cleanly you move them.

On NVIDIA GPUs, threads are grouped into warps. When threads in a warp access nearby memory addresses, the GPU can combine those loads and stores into a small number of memory transactions. This is called [**coalescing**](https://developer.nvidia.com/blog/unlock-gpu-performance-global-memory-access-in-cuda/).

Factorio translation:

> If 32 inserters grab 32 adjacent items from a clean belt, the belt is used efficiently. If those same 32 inserters each reach into a random chest somewhere else in the base, the logistics system still works, but throughput collapses.

This is why layout matters. Two kernels can perform the same number of mathematical operations and move the same number of logical elements, yet one can be much faster because the memory access pattern is cleaner.

In Factorio, spaghetti is funny until the factory needs to scale. In CUDA, spaghetti memory access is funny until the profiler tells you your bandwidth efficiency is bad.

## Registers, caches, and shared memory are just better chests

A naive Factorio player routes every item through the main bus. A better Factorio player adds local buffers. The assembler does not need to pull every single plate from the other side of the factory. Some items should sit right next to the subfactory that uses them. (To be fair this is not really true, you'll only really need buffers when you use trains in factorio, however for the sake of simplicity we'll assume you always need buffer)

GPUs have the same hierarchy:

- Registers are tiny and extremely fast.
- Shared memory and L1 cache are small but close to the compute units.
- L2 cache is larger and shared across more of the chip.
- HBM is huge and high-bandwidth, but far away compared with local storage.

The performance question is:

> How often do I have to go all the way back to the main bus?

If the answer is “every time I need anything,” your kernel is probably belt-bound. If the answer is “rarely, because I reuse local data,” you have a chance to keep the assemblers busy.

This is the reason behind many GPU optimization tricks:

- Use registers to keep values close.
- Use shared memory to stage tiles.
- Fuse kernels to avoid writing intermediate results to global memory.
- Prefer layouts that make access contiguous.
- Reuse data before evicting it.

In Factorio terms: stop shipping intermediate gears to a centralized storage if the next assembler is right there. (again centralized storage is more a thing in Satisfactory than factorio, but who cares...)

## Kernel fusion: stop shipping gears across the map

Suppose your factory makes gears in one subfactory, puts them on a train, ships them to a warehouse, unloads them, puts them on another belt, and feeds them into the next subfactory.

If the next subfactory is physically next door, this is insane.

Yet this is exactly what unfused GPU code can do. One kernel computes an intermediate tensor and writes it to global memory. The next kernel reads it back from global memory, performs a small operation, and writes another intermediate tensor. You are not doing much math. You are mostly moving temporary products between warehouses.

Kernel fusion says:

> Put the subfactories next to each other. Keep the intermediate item in hand. Do not send it back to the main bus unless you have to.

This is why fused attention, fused layernorm, fused activation kernels, and compiler systems like XLA, Triton, and torch.compile can matter so much. They are not just making the code prettier. They are removing unnecessary logistics.

## Batching: trains are expensive, fill them

Now we get to the part of the podcast that made the Factorio analogy really click for me.

In LLM inference, a forward pass requires reading model weights and doing computation. If you serve one user at a time, you may have to pay a huge memory cost to load weights just to produce one token for one sequence. That is like sending a giant train across the map to deliver supplies for one assembler cycle.

Batching changes the economics. Instead of using the weight fetch for one sequence, you use it for many sequences at once.

Factorio translation:

> The train was going to leave anyway. You might as well put more passengers on it.

This is why the podcast frames inference as a tradeoff between latency and cost. A larger batch can amortize weight loading over more tokens, making each token cheaper. But larger batches also mean each individual request may wait longer, and eventually other costs dominate.

The simplified Reiner equation for a forward pass has a memory term that includes both weight fetches and KV cache fetches:

```text
t_memory =  weight_fetch_time + KV_cache_fetch_time
```

The important asymmetry is:

- Weight fetches can be amortized across the batch.
- Compute cannot be amortized away, because each token needs work.
- KV cache reads cannot be amortized in the same way, because each sequence has its own context.

In Factorio terms, model weights are like a common ingredient shipment shared by many orders. The KV cache is more like each customer's private recipe history. You cannot share Alice's entire context with Bob.

## Where the analogy breaks

Like all analogies, this one is useful until it is not. I like this one because it explains a feeling I often have when learning systems: the math is simple once you know what to count, but knowing what to count is the hard part. Factorio trains exactly that instinct. You stop looking at the machine that is idle and start walking upstream until you find the first belt, inserter, station, or recipe that cannot keep up.

Factorio belts are visible and deterministic. GPU memory systems are hierarchical, concurrent, cached, scheduled, and full of hardware details that do not have perfect Factorio equivalents.

An assembler is not exactly a SM. A chest is not exactly a cache. A belt is not exactly HBM bandwidth. Factorio recipes do not branch. Belts do not perform speculative execution. Assemblers do not have register pressure. Trains do not have NUMA effects, unless your rail network is cursed enough that maybe they do.

But the analogy is still useful because it teaches the most important reflex:

> Do not ask only how much compute you have. Ask whether you can feed it.

That reflex transfers almost perfectly.

## The final mental model

If you remember only one thing from this post, remember this:

> A GPU bottleneck is often not about the number of assemblers. It is about the number of useful crafts per item delivered.

Or, less Factorio-poetically:

```text
performance ≤ min(compute throughput, memory bandwidth × operational intensity)
```

The factory must grow, but first, the belts must feed it.

## Further reading

- [Dwarkesh Patel with Reiner Pope: The math behind how LLMs are trained and served](https://www.dwarkesh.com/p/reiner-pope)
- [Reiner Pope flashcards from Dwarkesh](https://flashcards.dwarkesh.com/reiner-pope/)
- [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf)
- [NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [NVIDIA H100 Tensor Core GPU specifications](https://www.nvidia.com/en-us/data-center/h100/)
