# ADR-0002: Triton Inference Server Integration

## Status
Accepted

## Context
NIM runtime is built on Triton Inference Server. We need to decide how deeply to integrate with Triton's configuration and features versus abstracting them away.

## Decision
We will expose key Triton features through our CLI while providing sensible defaults and simplified configuration. Advanced users can still access full Triton configuration.

## Consequences

### Positive
- **Flexibility**: Users can leverage advanced Triton features when needed
- **Simplicity**: Beginners get working deployments with minimal configuration
- **Performance**: Direct access to batching, scheduling, and optimization features
- **Compatibility**: Works with existing Triton tooling and knowledge

### Negative
- **Complexity**: More configuration options can overwhelm new users
- **Maintenance**: Need to keep up with Triton API changes
- **Documentation**: Must explain both our abstractions and underlying Triton concepts

## Alternatives Considered

1. **Full Abstraction**: Hide all Triton details (too limiting)
2. **Pass-through Only**: Expose raw Triton config (too complex for beginners)
3. **Dual Interface**: Provide both simple and advanced modes (chosen approach)

## Implementation Notes
- Generate Triton model repository structure automatically
- Provide escape hatch for custom model.pbtxt files
- Support both model ensembles and single models
- Validate generated configurations before deployment