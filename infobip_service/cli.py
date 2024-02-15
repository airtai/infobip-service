__all__ = ["app"]

from pathlib import Path

import typer

app = typer.Typer()


@app.command()
def preprocess_dataset(
    input_path: Path = typer.Option(..., help="Path to the dataset to be preprocessed"),  # noqa
    output_path: Path = typer.Option(  # noqa
        ..., help="Path to the directory where the preprocessed dataset will be saved"
    ),
) -> None:
    """Preprocess the dataset."""
    from infobip_service.dataset.preprocessing import preprocess_dataset

    preprocess_dataset(input_path, output_path)


@app.command()
def preprocess_dataset_buckets(
    input_path: Path = typer.Option(..., help="Path to the dataset to be preprocessed"),  # noqa
    output_path: Path = typer.Option(  # noqa
        ..., help="Path to the directory where the preprocessed dataset will be saved"
    ),
) -> None:
    """Preprocess the dataset."""
    from infobip_service.dataset.preprocessing_buckets import preprocess_dataset

    preprocess_dataset(input_path, output_path)


@app.command()
def hello_from_cli() -> None:
    """Prints hello from the CLI."""
    typer.echo("Hello from the CLI!")
