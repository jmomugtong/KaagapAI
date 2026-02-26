# Contributing to MedQuery

Thank you for your interest in contributing to MedQuery! This guide will help you get started.

## Getting Started

1. **Fork** the repository
2. **Clone** your fork:
   ```bash
   git clone https://github.com/<your-username>/MedQuery.git
   cd MedQuery
   ```
3. **Set up** the development environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -e ".[dev,test]"
   ```
4. **Start services**:
   ```bash
   docker-compose up -d
   ```

## Development Workflow

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feat/your-feature
   ```
2. Write tests first (TDD approach)
3. Implement your changes
4. Ensure all checks pass:
   ```bash
   make check    # lint + format-check + type-check
   make test     # run test suite
   ```
5. Commit with a descriptive message (see below)
6. Push and open a Pull Request

## Code Style

- **Formatter**: Black (line length 88)
- **Import sorting**: isort (black profile)
- **Linter**: Ruff
- **Type checking**: mypy (strict mode)

Run all checks at once:
```bash
make check
```

Format code:
```bash
make format
```

## Testing

- Minimum **80% code coverage** required
- Use pytest markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`
- Run tests:
  ```bash
  make test          # all tests
  make test-unit     # unit tests only
  make test-cov      # with coverage report
  ```

## Commit Messages

Follow the conventional commits format:

```
type(scope): short description

Optional longer description.
```

**Types**: `feat`, `fix`, `perf`, `docs`, `test`, `refactor`, `config`, `ci`

**Examples**:
- `feat(retrieval): Add medical synonym expansion`
- `fix(parser): Handle missing confidence in LLM output`
- `perf(llm): Add stop sequences to reduce token waste`
- `docs: Update API examples in README`

## Pull Request Guidelines

- Keep PRs focused — one feature or fix per PR
- Include tests for new functionality
- Update documentation if behavior changes
- Ensure CI passes before requesting review
- Link related issues in the PR description

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include steps to reproduce for bugs
- Include expected vs actual behavior
- Attach relevant logs or screenshots

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
