"""Command line interface for the project."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from .anomaly import RobustZScoreDetector
from .config import RuntimeSettings
from .reporting import build_report
from .simulation import AnomalyInjection, SensorSimulator, SensorSpec, SimulationConfig
from .streaming import InMemoryAlertSink, StreamingConfig, StreamProcessor, iter_readings

app = typer.Typer(help="Portfolio project command line interface.")
console = Console()


@app.command()
def info() -> None:
    """Print validated runtime settings."""
    settings = RuntimeSettings()
    console.print(settings.model_dump())


@app.command()
def simulate(
    steps: Annotated[int, typer.Option(help="Number of time steps to generate.")] = 1_440,
    seed: Annotated[int, typer.Option(help="Random seed for reproducibility.")] = 42,
    output: Annotated[
        Path | None, typer.Option(help="Optional CSV path to write the dataset.")
    ] = None,
) -> None:
    """Generate a synthetic multivariate sensor dataset and summarize it."""
    config = SimulationConfig(
        sensors=[
            SensorSpec(sensor_id="temperature", baseline=60.0, daily_amplitude=8.0, unit="C"),
            SensorSpec(sensor_id="vibration", baseline=0.4, noise_std=0.05, unit="g"),
            SensorSpec(sensor_id="pressure", baseline=101.0, trend_per_day=0.2, unit="kPa"),
        ],
        n_steps=steps,
        seed=seed,
    )
    frame = SensorSimulator(config).run()
    console.print(f"Generated {len(frame)} rows across {frame['sensor_id'].nunique()} sensors.")
    if output is not None:
        frame.to_csv(output, index=False)
        console.print(f"Wrote dataset to {output}")


@app.command()
def stream(
    steps: Annotated[int, typer.Option(help="Number of time steps to simulate.")] = 720,
    seed: Annotated[int, typer.Option(help="Random seed for reproducibility.")] = 42,
) -> None:
    """Simulate a stream, detect anomalies online, and print a report."""
    config = SimulationConfig(
        sensors=[
            SensorSpec(sensor_id="temperature", baseline=60.0, daily_amplitude=8.0, unit="C"),
            SensorSpec(sensor_id="vibration", baseline=0.4, noise_std=0.05, unit="g"),
        ],
        n_steps=steps,
        seed=seed,
        anomalies=[
            AnomalyInjection(sensor_id="temperature", start_step=steps // 2, magnitude=30.0),
            AnomalyInjection(
                sensor_id="vibration", start_step=steps // 3, duration=20, magnitude=0.5,
                kind="drift",
            ),
        ],
    )
    frame = SensorSimulator(config).run().sort_values("timestamp")

    sink = InMemoryAlertSink()
    processor = StreamProcessor(
        RobustZScoreDetector(window_size=60, threshold=4.0),
        sink,
        config=StreamingConfig(window_size=400, min_window=80, batch_size=50),
    )
    rows = zip(frame["sensor_id"], frame["timestamp"], frame["value"], strict=True)
    total = processor.run(iter_readings(rows))

    console.print(f"Processed {len(frame)} readings; emitted {total} alerts.")
    console.print(build_report("Streaming run", sink.alerts, top_n=5))


if __name__ == "__main__":
    app()
