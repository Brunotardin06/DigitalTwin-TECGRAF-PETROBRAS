import click

from src.inputs.config import ValidationRunRequest
from src.workflow import SimulationController


@click.command()
@click.option("--config-name", default="main")
@click.option("--skip-animation", is_flag=True, default=False)
@click.option("--skip-simulation", is_flag=False, default=False)
@click.option("--skip-plots", is_flag=True, default=False)
@click.option("--evaluation", is_flag=True, default=False)
@click.option("--optimize-wdf-cdf", is_flag=True, default=False)
@click.option("--fast-particles-per-wdf", type=int, default=1)
@click.option("--wdf-min", type=float, default=0.0)
@click.option("--wdf-max", type=float, default=0.05)
@click.option("--wdf-step", type=float, default=0.0025)
@click.option("--cdf-min", type=float, default=0.5)
@click.option("--cdf-max", type=float, default=1.0)
@click.option("--cdf-step", type=float, default=0.1)
@click.option("--padding-animation-frame", type=float, default=0.1)
@click.option("--wind-drift-factor", type=float, default=None)
@click.option("--current-drift-factor", type=float, default=None)
@click.option(
    "--processes-dispersion",
    type=click.Choice(["true", "false"], case_sensitive=False),
    default=None,
)
@click.option(
    "--processes-evaporation",
    type=click.Choice(["true", "false"], case_sensitive=False),
    default=None,
)
@click.option("--oil-types", type=str, default=None)
@click.option("--oil-types-file", type=str, default=None)
@click.option("--environment", type=str, default="2019")
@click.option("--shp-zip", type=str, default=None)
@click.option("--min-long", type=float, default=None)
@click.option("--max-long", type=float, default=None)
@click.option("--min-lat", type=float, default=None)
@click.option("--max-lat", type=float, default=None)
@click.option("--start-index", type=int, default=0)
@click.option("--environmental-offset-hours", type=float, default=None)
@click.option("--environmental-offset-values", type=str, default=None)
def simulate_validation(**kwargs):
    offset_values = kwargs.pop("environmental_offset_values", None)
    if offset_values:
        kwargs["environmental_offset_values"] = [
            float(item.strip())
            for item in offset_values.split(",")
            if item.strip()
        ]
    controller = SimulationController()
    request = ValidationRunRequest(**kwargs)
    controller.run_validation(request)


if __name__ == "__main__":
    simulate_validation()
