# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for the nimify-anything project. ADRs document important architectural decisions, their rationale, and their impact on the system.

## ADR Template

Each ADR should follow this structure:

```markdown
# ADR-XXXX: [Title]

## Status
[Proposed | Accepted | Deprecated | Superseded]

## Context
Brief description of the architectural decision and why it needs to be made.

## Decision
What is the decision being made?

## Consequences
What are the positive and negative consequences of this decision?

## Alternatives Considered
What other options were evaluated?
```

## Index

- [ADR-0001: Use NVIDIA NIM Runtime as Backend](./0001-use-nvidia-nim-runtime.md)
- [ADR-0002: Triton Inference Server Integration](./0002-triton-inference-server.md)
- [ADR-0003: Kubernetes-First Deployment Strategy](./0003-kubernetes-first-deployment.md)
- [ADR-0004: Prometheus Metrics Collection](./0004-prometheus-metrics.md)
- [ADR-0005: OpenAPI Schema Generation](./0005-openapi-schema-generation.md)

## Creating New ADRs

1. Create a new file with format `XXXX-brief-title.md`
2. Use the next sequential number
3. Follow the template structure
4. Add an entry to the index above
5. Submit for review via pull request