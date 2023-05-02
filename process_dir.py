from pathlib import Path

import click
from main import FaceAnonymizer


@click.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False))
def process_dir(path: str):
    face_blur = FaceAnonymizer()
    out_path = face_blur.process_dir(Path(path))
    click.launch(str(out_path))


if __name__ == "__main__":
    process_dir()
