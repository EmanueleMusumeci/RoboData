---
applyTo: '.py'
---
When running python scripts, use the full module path to ensure the correct context is set. For example, if you are in the `RoboData` directory, run:

ENVIRONMENT_ACTIVATION && python3 -m core ...

where:
- ENVIRONMENT_ACTIVATION is the command to activate your virtual environment. In our case the virtual environment is activated by running `. .venv/bin/activate` in the root RoboData directory.
- `core` is the main module of the project and ... represents the specific script or module you want to execute. Most