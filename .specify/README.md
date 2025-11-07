# Spec-Kit Documentation

This directory contains the Spec-Kit specifications and artifacts for the Speckit LLM Evidence project.

## Structure

```
.specify/
├── memory/
│   └── constitution.md          # Project principles and guidelines
├── specs/
│   └── 01-extractive-qa-system.md  # Extractive QA system specification
├── artifacts/                    # Generated artifacts (gitignored)
└── README.md                     # This file
```

## Constitution

The [constitution.md](memory/constitution.md) file contains the core principles and guidelines for the project:

- **Mission**: Fine-tune extractive QA for criteria matching
- **Core Principles**: Reproducibility, performance optimization, data integrity, etc.
- **Non-Negotiables**: GPU requirement, MLflow tracking, format consistency
- **Decision Framework**: Prioritization guidelines for technical decisions

## Specifications

### [01-extractive-qa-system.md](specs/01-extractive-qa-system.md)

Detailed specification for the extractive QA system including:

- **Data Pipeline**: Dataset handling, preprocessing, loading
- **Model Architecture**: Gemma-based encoder with QA heads
- **Training Pipeline**: Optimization, loss functions, RTX 4090 optimizations
- **Evaluation**: Metrics (EM, F1), evaluation scripts
- **MLflow Integration**: Tracking, artifacts, model registry
- **Logging**: Structured logging and visualization
- **Configuration**: YAML configs for all components

## Spec-Kit Commands

These commands are available through the spec-kit CLI (when installed):

- `/speckit.clarify` - Resolve specification gaps
- `/speckit.analyze` - Check artifact consistency
- `/speckit.checklist` - Validate requirements quality
- `/speckit.specify` - Define what to build
- `/speckit.plan` - Outline technical approach
- `/speckit.tasks` - Break down work into tasks
- `/speckit.implement` - Build the feature

## Usage

1. **Review the constitution** to understand project principles
2. **Read specifications** before implementing features
3. **Keep specs updated** as the project evolves
4. **Use spec-kit commands** to maintain alignment

## Validation

Specifications are validated in CI/CD via GitHub Actions:

- `.github/workflows/spec-validation.yml` - Validates spec-kit structure
- `.github/workflows/code-quality.yml` - Validates configs and code

## Contributing

When adding new features:

1. Update relevant specifications in `specs/`
2. Ensure changes align with `constitution.md`
3. Run spec validation: `specify analyze`
4. Update this README if adding new specs
