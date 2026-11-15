"""Model tournament: compare candidate architectures under identical data / token budget / batch, measuring quality,
speed and memory. plan.py + report.py are plain Python (no torch); check.py + probe.py need torch; runner.py launches
one training process per free GPU. Everything is driven by a plan file in configs/tournament/."""
