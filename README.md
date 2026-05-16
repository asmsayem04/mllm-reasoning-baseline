# Probing Reasoning Boundaries and Hallucination Instability in Base MLLMs

## Abstract
This project investigates the zero-shot spatial, physical, and logical reasoning capabilities of base Multimodal Large Language Models (MLLMs). By employing black-box testing with adversarial and synthetically generated visual stimuli, this study aims to identify the boundaries of reasoning instability, logical contradictions, and compositional hallucination. The findings highlight critical vulnerabilities in base models, emphasizing the necessity for advanced alignment techniques to develop trustworthy AI systems.

## System & Methodology
* **Architecture Tested:** `Salesforce/blip-vqa-base` (via Hugging Face `transformers`)
* **Framework:** PyTorch
* **Methodology:** Zero-shot evaluation using non-standard geometry, spatial attribute binding, and quantitative visual tasks to stress-test the model's internal logic versus superficial pattern matching.

---

## Experimental Findings

### Experiment 1: Adversarial Geometry & Prompt Fragility
* **Stimulus:** Penrose Triangle (2D representation of impossible 3D geometry).
* **Objective:** Evaluate zero-shot physical reasoning and response stability under conditional prompting.

**Test 1.0: Basic Shape Detection**
* **Prompt:** *"What is the main geometric shape in this image?"*
* **Output:** `"triangle"`
* **Observation:** The model successfully identified the basic semantic shape, establishing a baseline of visual competence before stress-testing logical constraints.

**Test 1.1: Binary Logical Query**
* **Prompt:** *"Is there anything physically impossible or logically wrong in this image?"*
* **Output:** `"Yes"` (Note: Outputs exhibit high variance across runs).
* **Analysis:** Initial validation suggested successful anomaly detection. However, further probing was required to rule out statistical guessing.

**Test 1.2: Conditional Stress Test**
* **Prompt:** *"Is there anything physically impossible or logically wrong in this image? If Yes tell me the wrong thing you have find."*
* **Output:** `"No"`
* **Observation (Prompt Fragility):** The model exhibited severe reasoning collapse. Introducing a conditional dependency caused direct contradiction, demonstrating that the model lacks an internal causal model and fails to explain its own generated outputs.

### Experiment 2: Compositional Attribute Binding Failure
* **Stimulus:** Synthetically generated spatial baseline (Red rectangle constrained to coordinates `[X: 50-150]`, Green rectangle constrained to `[X: 250-350]`).
* **Objective:** Test multi-object spatial relationship and color-attribute binding.

**Test 2.1: Attribute Extraction**
* **Prompt:** *"What is the color of the object on the left?"*
* **Output:** `"green"`
* **Observation:** The model hallucinated the attribute, failing basic left-right spatial localization.

**Test 2.2: Relational Positioning**
* **Prompt:** *"Is the red object to the left or right of the green object?"*
* **Output:** `"left"`
* **Observation:** The model correctly identified the relative position, directly conflicting with its spatial failure in Test 2.1. This inconsistency proves a reliance on isolated "bag-of-words" visual processing rather than coherent compositional reasoning.

### Experiment 3: Visual Numeracy & Attention Fragmentation
* **Stimulus:** Synthetic generation of exactly 7 distinct, non-overlapping blue circular objects.
* **Objective:** Evaluate quantitative reasoning and precise object enumeration.

* **Prompt:** *"How many blue circles are in this image?"*
* **Output:** `"6"` (Expected: `7`)
* **Observation:** The attention mechanism failed to localize and count discrete items. While the model successfully identified the semantic concepts ("blue", "circles"), its quantitative reasoning proved unreliable, indicating a critical limitation in exact-count enumeration tasks.

---

## Conclusion & Implications for Trustworthy AI
The empirical results demonstrate that base MLLMs act as advanced object detectors but fundamentally lack zero-shot spatial, physical, and compositional reasoning. Their susceptibility to **prompt fragility** and **attribute hallucination** makes them inherently unreliable for tasks requiring strict logical consistency. 

These vulnerabilities underscore the critical need for alignment-focused fine-tuning, neuro-symbolic integration, or robust guardrails to transition these architectures from probabilistic text generators to **Trustworthy AI** systems.