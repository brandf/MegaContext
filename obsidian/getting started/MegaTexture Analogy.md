---
tags:
  - gettingstarted
summary: MegaContext is inspire by graphics tech called MegaTexture
---
## Analogy: MegaTexture → MegaContext

- In graphics, **MegaTexture** streams the visible portions of a vast texture mipmap into GPU memory at the appropriate resolution.
- **MegaContext** mirrors that idea for language: instead of mipmap tiles, it maintains embeddings at multiple levels of detail (token L0, gist L1, gist L2, …), yielding effectively unbounded context for a frozen LLM.

---

For those that are interested in learning about the inspiration, [this video](https://www.youtube.com/watch?v=BiQCz2NjPR8) provides a good overview of the problems Mega Texture solves.

---

## What does MegaTexture offer?

1. an **indirection** between the full (precomputed) texture mipmap trees on disk, and the mixed-LOD subset on the GPU to support a fixed sized render target
    you only need roughly enough texture pixels on GPU to cover the screen at the ideal LODs

2. **incrementally streaming** in changes to the 'working set' of LOD chunks from disk to GPU - as a function of what's visible on screen
    as you look around, detail is evicted and more relevant detail is streamed in, but frame over frame not much changes.

3. users/artists don't have to worry about texture size as much, and can basically have **as much detail as they want** because disk >> ram
    there is still a practical limit, but not loading all detail up front gives you much more bang for your buck.

  ---

## [[How MegaContext Works|How do MegaTexture concepts map to MegaContext]]?

1. instead of just one context, there is an **indirection** between a [[MegaContext Tree]] (potentially on disk) and a [[Working Context]] on the GPU (like a typical LLM context but included mixed-LOD gists latents LOD1+ as well as LOD0 tokens)
    The [[Working Context]] is a fixed size (entries) smaller context, whereas the [[MegaContext Tree]] is a long unbounded context (well gist tree)

    These are 1d context rather than 2d things like mipmaps, but you can think of a gist as a learned latent space summary of a span of tokens

2. as the context grows a learned '[[LensNet]]' decides which [[Working Context]] entries it wants more/less detail on, and **incrementally streams** in the changes.
    I call it a Lens because it's basically focusing all of the [[MegaContext Tree]]'s worth of 'time' onto a fixed sized [[Working Context]], using gists to compress parts to make it fit.

    The [[LensNet]] learns the policy of what information is relevant, conditioned on the current query (recent context). This means things that were in-focus can become out of focus if the conversation evolves in a different direction. The defocusing addresses a common problem with 'distractors' in the context, which are similar to 'aliasing' in a texture without mipmaps.

3. the end result, if I get it working, should be a system that can 'wrap' any model with some light fine-tuning, and upgrade it to have **effectively infinite context lengths** with sub-linear compute/memory
    this could open up all kinds of new scenarios like having entire codebases maintained as a system-prompt-like-[[MegaContext Tree]] that is updated any time a file changes
