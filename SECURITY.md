# Security Policy

## Supported Versions

This project is pre-`1.0.0`. Security fixes target the latest release and the `main` branch.

| Version | Supported |
| --- | --- |
| 0.1.x | Yes |

## Reporting a Vulnerability

Please do not open public issues for suspected vulnerabilities.

Report security concerns privately by emailing the maintainer listed in `pyproject.toml`.
Include:

- affected version or commit
- dependency, endpoint, or module involved
- reproduction steps or proof of concept
- expected impact
- any suggested mitigation

You can expect an initial acknowledgement within 7 days. Confirmed vulnerabilities will be
fixed in the smallest practical change and documented in the changelog or release notes.

## Dependency Security

Default runtime dependencies are audited with `pip-audit` during release work. GitHub
Dependabot is configured to monitor Python dependencies and GitHub Actions.
