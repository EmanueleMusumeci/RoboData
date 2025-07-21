---
applyTo: '**'
---
You can use the main.py script as a module. First you need to activate your virtual environment then you can run the module. In the root RoboData directory, run:

```bash
. .venv/bin/activate && python -m backend.main
```

To run with a query use the -q option:

```bash
. .venv/bin/activate && python -m backend.main -q "Your query here"
```