# Contributing

## Contributing a New Deep Personalization Benchmark

We accept contributions that expand the repository with well-defined, reproducible
benchmarks for personalizing AI agent behavior with rich user observation or 
interaction data. We recommend opening an issue before beginning a large benchmark 
contribution so its scope and integration can be discussed early.

A new benchmark should include:

- An implementation under `benchmarks/<benchmark_name>/` that follows the shared
  prepare, predict, and score workflow where applicable.
- A README describing the task, dataset and split structure, metrics, setup, and
  complete evaluation commands.
- Clear data provenance, download instructions, and licensing terms. Do not include
  private or sensitive data, or any data you do not have permission to redistribute.
- Deterministic example identifiers and reproducible evaluation and scoring logic.
- Tests covering data validation, core metrics, and the primary evaluation path.
- Any required CLI entry points and package-data declarations in `pyproject.toml`.

When possible, include a simple baseline or expected result so maintainers can verify
the benchmark end to end.

## Development Checks

Install the development dependencies and run the checks relevant to your change:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Signing Off Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO (https://developercertificate.org/):

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

    Everyone is permitted to copy and distribute verbatim copies of this
    license document, but changing it is not allowed.


    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I
        have the right to submit it under the open source license
        indicated in the file; or

    (b) The contribution is based upon previous work that, to the best
        of my knowledge, is covered under an appropriate open source
        license and I have the right under that license to submit that
        work with modifications, whether created in whole or in part
        by me, under the same open source license (unless I am
        permitted to submit under a different license), as indicated
        in the file; or

    (c) The contribution was provided directly to me by some other
        person who certified (a), (b) or (c) and I have not modified
        it.

    (d) I understand and agree that this project and the contribution
        are public and that a record of the contribution (including all
        personal information I submit with it, including my sign-off) is
        maintained indefinitely and may be redistributed consistent with
        this project or the open source license(s) involved.
  ```
