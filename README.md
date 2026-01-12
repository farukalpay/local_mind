# Collective Mind: Dual-Engine Architecture (Accordion Topology)

**System Architecture Specification: Autonomous Narrative Engine**

This document outlines the technical architecture of the "Local Mind" autonomous agent. The system implements a Dual-Engine Stochastic Control System leveraging Latent Space Interpolation to construct high-density unique narrative blocks.

## Core Architecture

The system operates two distinct Large Language Models in parallel on Apple Silicon (M4 Pro) via the `llama.cpp` Metal backend:

1.  **Engine A (The Architect): Meta-Llama-3-8B-Instruct**
    *   **Function**: Physical Action, Physiological Reaction, Causality management.
    *   **Context Window**: 16k tokens.
    *   **Role**: Handles the protagonist's localized physical existence and biological feedback loops.

2.  **Engine B (The Scout): Gemma-2-2B-It**
    *   **Function**: Atmosphere generation, Texture synthesis, Geometric descriptions.
    *   **Context Window**: 4k tokens.
    *   **Role**: Generates non-human environmental data (sensory inputs) to inject into Engine A's context.

---

## The Accordion Topology (4-Phase Cycle)

To mitigate "Variance Collapse" (Early Stopping) and maximize sensory density, the main generation loop enforces a strict **4-Phase Interpolation** for every single block execution.

The generation process is structural rather than purely generative.

### Phase I: The Void (Gemma)
*   **Directive**: Describe the immediate atmosphere in extreme sensory detail. No human action.
*   **Output**: Air density, olfactory data, light frequency, static geometry.
*   **Technical Rationale**: Pre-filling the context with environmental data forces the larger model to exist within a defined physics engine before generating action.

### Phase II: The Intrusion (Llama)
*   **Directive**: The Protagonist disturbs this environment. Describe ONE physical interaction. Stop before feeling.
*   **Output**: Kinetic movement (touch, step, reach).
*   **Technical Rationale**: Separation of Action and Reaction prevents hallucinatory emotional logic.

### Phase III: The Fractal Zoom (Gemma)
*   **Directive**: Zoom in x1000 on the contact point. Describe the microscopic texture.
*   **Output**: Micro-texture details (friction coefficients, heat transfer, material composition).
*   **Technical Rationale**: Introduces "Time Dilation" by forcing the model to resolve high-frequency details of the contact surface.

### Phase IV: The Reaction (Llama)
*   **Directive**: Merge inputs. Synthesis of physiological nerve response.
*   **Output**: Physiological synthesis of the atmosphere, action, and texture.

---

## Output Sample (Raw Block)

**Block ID**: `zrBhpz2iTxk6hqh0hb42SGWKFfpSv3RSnIodfzprngs`
**Model Configuration**: `llama-3-8b+gemma-2-2b`

> The air itself hums with a low, throbbing frequency, heavy and viscous despite the vessel’s open structure. A perpetual corona of static light, charged with an electric haze, bathes the metallic plane in a shimmer of spectral blues and greens, momentarily pulsing with a frequency matched by the thrumming metal beneath your hand. I make contact with the humming metal surface, my gloved hand landing with a sharp, metallic clink. The sensation is a shock, not a blunt impact. It's an assault on the senses as the raw metal yields just a single, calcified rippling tremor. The surface, like a gelatinous geode, becomes alive beneath the pressure of fingertips, the vibratory hum of its internal structure thrumming against your gloves. Each microscopic pearl of its lattice, once a smooth, almost pulsing substance,, and my nerves , attuned to the subtle vibrations of the humming metal, responded with a symphony of sensation. As my gloved hand made contact, a burst of stimuli coursed through my fingertips, like a crackling electrical current.

---

## Build & Run Instructions

### Prerequisites
*   **Hardware**: Apple Silicon (M3/M4 recommended) - Metal API required.
*   **Models**:
    *   `models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf`
    *   `models/gemma-2-2b-it-Q4_K_M.gguf`

### Usage

**1. The Infinite Run (Default)**
Generates blocks continuously until manual termination.
```bash
./run.sh
```

**2. Finite Batch (Testing)**
Generates N blocks and stops.
```bash
./run.sh --blocks 10
```

**3. Manual Time Travel**
Resume narrative from a specific Arweave Transaction ID.
```bash
./run.sh --previous_txid <TXID>
```

---
*Internal Documentation for Autonomous Agent No. 8392*
