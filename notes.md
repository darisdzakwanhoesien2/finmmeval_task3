Workflow and Implementation

https://scite.ai/assistant/you-are-an-academic-researcher-writing-a-master-s-thesis-thesis-Pz1QXV


\section{Workflow}

Overview of the Task 3 execution cycle

The research workflow operates on a 00:00 UTC trigger sequence: Request Reception -> News/Price Synthesis -> Reasoning -> Action Return -> Position Execution at Close Price. This end-to-end daily cycle forms the backbone of the empirical Arena-style evaluation, enabling deterministic turnover and auditable decision provenance within fixed latency constraints (Qian, 2025; , Singhi, 2025; , Yu, 2025; , Blanchard, 2025).
Daily data ingestion and synthesis

Each day, a standardized input bundle is assembled from real-time signals (sentiment milestones, momentum indicators, and price histories) along with regulatory objects where available. Synthesized into a coherent context, this bundle feeds the reasoning module to produce a discrete Action_t (BUY, HOLD, SELL) for immediate execution at close, preserving the semantic alignment between perception and action under time pressure (Jadhav & Mirza, 2025; , Khak et al., 2025; , Fu, 2025; , Qian, 2025; .
Reasoning and decision generation

The reasoning stage integrates neurosymbolic inference with rule-based thresholds to map multi-modal inputs to a single actionable decision. The output is a JSON payload containing the recommended_action and accompanying provenance, including source citations and numeric evidences, suitable for auditable governance reviews Chen et al., 2018; , Rahman, 2025).
End-to-end execution and logging

The Actiont is transmitted to the execution layer, which performs turnover by replacing the prior day’s Position{t-1} with Position_t, and records latency, token usage, and decision provenance. Execution occurs at or near the close price, aligning with the designated daily window and ensuring reproducible end-of-day positioning for the next cycle (Long, 2025; , Qian, 2025; , Qian, 2025).
Grounding, provenance, and governance

Throughout the workflow, grounding strategies (RAG, metadata contexts, verified numerics) ensure outputs are anchored to verifiable documents and data points. Decision provenance includes citations, timestamps, and rationale traces to support post-hoc governance checks and regulatory scrutiny, reinforcing the integrity of automated trading decisions in high-stakes domains (Jadhav & Mirza, 2025; , Khak et al., 2025; , Fu, 2025; , Chen et al., 2018; , Tan, 2025).
Asset-specific considerations and cross-asset alignment

The architecture supports TSLA and BTC as primary test assets, with asset-specific grounding pipelines reflecting the presence or absence of formal filings. The workflow accommodates regulatory diversity while maintaining a unified API and evaluation framework, enabling cross-asset comparisons and robustness checks under identical latency budgets and turnover rules (Nguyễn, 2025; , Amin et al., 2022; , Jaiswal et al., 2024; , Qian, 2025).

\subsection{Implementation (Revised)}

Endpoint deployment and action schema

The trading endpoint is deployed behind a FastAPI-based server, exposing a JSON API that returns "recommendedaction": { "BUY": boolean, "HOLD": boolean, "SELL": boolean } along with ancillary metadata (confidence, rationale references, and provenance). The endpoint enforces a deterministic turnover policy where the new Actiont replaces Position{t-1} to produce Positiont, ensuring a single active signal at close (Fu, 2025; , Yu, 2025).
Data ingestion pipeline

Daily JSON inputs aggregating sentiment milestones, momentum indicators, and price history are parsed by the ingestion service, validated against a versioned schema, and forwarded to the reasoning module within the 3-minute latency envelope. The pipeline supports retrieval-augmented grounding and provenance tagging to maintain auditable linkages between inputs and outputs (Jadhav & Mirza, 2025; , Khak et al., 2025; , Fu, 2025; , Chen et al., 2018).
Reasoning and action mapping

The Reasoning module consumes structured inputs, performs neurosymbolic inference, and outputs Action_t via a deterministic policy π: E × T → {BUY, HOLD, SELL}. Each action is accompanied by a concise justification and source citations, enabling governance reviews and regulatory traceability Chen et al., 2018), Rahman, 2025).
Turnover and execution layer

The Execution layer applies Actiont by updating Positiont to reflect the exposure implied by the action, discarding any prior residual signals. Turnover costs, including slippage and fees, are computed and logged, allowing end-to-end assessment of turnover impact on CR and SR under fixed deadlines (Long, 2025; , Qian, 2025; , Qian, 2025).
Grounding, provenance, and security

Grounding relies on RAG with metadata context and numerically verified inferences; provenance captures source documents, timestamps, and evidence embeddings. Security controls enforce access authentication, data privacy, and audit trails suitable for regulatory review (Jadhav & Mirza, 2025; , Khak et al., 2025; , Fu, 2025; , Chen et al., 2018), Tan, 2025).
Asset grounding policies

For TSLA, grounding leverages formal 10-K/Q disclosures and governance signals; for BTC, grounding relies on credible external analyses and on-chain indicators, with asset-specific provenance schemas to sustain cross-asset comparability (Nguyễn, 2025; , Amin et al., 2022; , Jaiswal et al., 2024).
Logging and reproducibility

All trials log input provenance, API call details, latency, Actiont, Positiont, turnover costs, and decision provenance traces. Prompt templates and versioning are tracked to enable reproducible experiments across laboratories within the Task 3 framework (Khak et al., 2025; , Rahman, 2025; , Lee et al., 2024; , Fu, 2025; , Chen et al., 2018).

