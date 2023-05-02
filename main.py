import math
import random
from pathlib import Path
from itertools import chain

import cv2
import click

from mediapipe.python.solutions import face_detection
from mediapipe.python.solutions import drawing_utils

random.seed(42)

video_extensions = [".mp4", ".avi", ".mov"]
image_extensions = [".jpg", ".jpeg", ".png"]
supported_extensions = video_extensions + image_extensions


class FaceAnonymizer:
    def __init__(
        self,
        confidence: float = 0.5,
        face_min_size: int = 5,
        face_expand: float = 0.2,
        pixelation_factor: int = 20,
        window_sizes: int = (400, 700),
        local_progress_callback=None,
        global_progress_callback=None,
    ) -> None:
        self.detector = face_detection.FaceDetection(
            min_detection_confidence=confidence, model_selection=1
        )

        self.face_min_size = face_min_size
        self.face_expand = face_expand
        self.pixelation_factor = pixelation_factor
        self.window_sizes = window_sizes

        self.local_progress_callback = local_progress_callback
        self.global_progress_callback = global_progress_callback

    def pixelate(self, img, x1, y1, x2, y2):
        # Pixelate the face
        face = img[y1:y2, x1:x2]
        pixelated_size = (
            int((x2 - x1) / self.pixelation_factor),
            int((y2 - y1) / self.pixelation_factor),
        )
        # Check if the pixelation is too small
        if pixelated_size[0] < 1 or pixelated_size[1] < 1:
            return img

        face = cv2.resize(face, pixelated_size)
        face = cv2.resize(face, (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA)
        img[y1:y2, x1:x2] = face

    def process_img(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.detector.process(img_rgb)
        if faces.detections:
            for detection in faces.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = img.shape

                # Calculate the bounding box
                x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
                x2, y2 = int((bbox.xmin + bbox.width) * w), int(
                    (bbox.ymin + bbox.height) * h
                )

                # Make the bounding box bigger
                x1 = x1 - int(self.face_expand * (x2 - x1))
                y1 = y1 - int(self.face_expand * (y2 - y1))
                x2 = x2 + int(self.face_expand * (x2 - x1))
                y2 = y2 + int(self.face_expand * (y2 - y1))

                # Check if the bounding box is within the image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Check if the bounding box is too small
                if x2 - x1 < self.face_min_size or y2 - y1 < self.face_min_size:
                    continue

                # drawing_utils.draw_detection(img, detection)
                # cv2.imshow("face", img)
                # cv2.waitKey(0)
                yield (x1, y1, x2, y2)

    def process(self, img):
        w, h, _ = img.shape
        pixelated_size = (
            int(h / self.pixelation_factor),
            int(w / self.pixelation_factor),
        )

        pixelated_img = cv2.resize(img, pixelated_size)
        pixelated_img = cv2.resize(pixelated_img, (h, w), interpolation=cv2.INTER_AREA)

        for x1, y1, x2, y2 in chain(self.sliding_windows(img), self.process_img(img)):
            # random_color = (
            #     random.randint(0, 255),
            #     random.randint(0, 255),
            #     random.randint(0, 255),
            # )
            # cv2.rectangle(img, (x1, y1), (x2, y2), random_color, 2)

            img[y1:y2, x1:x2] = pixelated_img[y1:y2, x1:x2]
        return img

    def sliding_windows(self, img):
        for window_size in self.window_sizes:
            yield from self.sliding_window(img, window_size)

    def sliding_window(self, img, window_size):
        w, h, _ = img.shape

        window_step = window_size / 2

        windows_x = math.ceil(w / (window_step))
        windows_y = math.ceil(h / (window_step))

        step_x = int(w / windows_x)
        step_y = int(h / windows_y)

        for i in range(windows_x):
            x1 = int(i * step_x)
            x2 = int((i + 1) * step_x)
            for j in range(windows_y):
                y1 = int(j * step_y)
                y2 = int((j + 1) * step_y)

                # random_color = (
                #     random.randint(0, 255),
                #     random.randint(0, 255),
                #     random.randint(0, 255),
                # )
                # cv2.rectangle(img, (y1, x1), (y2, x2), random_color, 2)

                window = img[x1:x2, y1:y2]
                for fx1, fy1, fx2, fy2 in self.process_img(window):
                    yield (y1 + fx1, x1 + fy1, y1 + fx2, x1 + fy2)

    def process_image_file(self, in_path: Path, out_path: Path):
        img = cv2.imread(str(in_path))
        img = self.process(img)
        cv2.imwrite(str(out_path), img)

    def process_video_file(self, in_path: Path, out_path: Path):
        capture = cv2.VideoCapture(str(in_path))
        out = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            capture.get(cv2.CAP_PROP_FPS),
            (
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            ),
        )

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        with click.progressbar(
            length=total_frames,
            label="Processing video frames",
            show_pos=True,
        ) as bar:
            while capture.isOpened():
                read, frame = capture.read()
                if not read:
                    break

                img = self.process(frame)
                out.write(img)
                bar.update(1)
                if self.local_progress_callback:
                    self.local_progress_callback(bar.pct)

        capture.release()
        out.release()

    def _process_file(self, filename: Path, out_filepath: Path, global_progress=0):
        if filename.suffix in video_extensions:
            out_filepath = out_filepath.with_suffix(".mp4")
            filetype = "video"
        else:
            filetype = "image"

        if self.global_progress_callback:
            self.global_progress_callback((global_progress, filetype, filename))

        click.echo(click.style(f"Processing {filetype} {filename}", fg="blue"))

        if filetype == "video":
            self.process_video_file(filename, out_filepath)
        else:
            self.process_image_file(filename, out_filepath)

        click.echo(click.style(f"Saved {filetype} to {out_filepath}", fg="green"))

    def process_file(self, filename: Path):
        out_filename = filename.with_stem(f"{filename.stem}_anonymized")
        self._process_file(filename, out_filename)

    def process_dir(self, path: Path):
        out_path = path.parent / f"{path.stem}_anonymized"
        files_to_process = list(path.rglob("*"))
        files_to_process = [
            f
            for f in files_to_process
            if f.is_file() and f.suffix in supported_extensions
        ]
        files_to_process = sorted(
            files_to_process, key=lambda f: f.suffix in image_extensions, reverse=True
        )

        with click.progressbar(files_to_process, label="Processing files") as bar:
            for filename in bar:
                out_subpath = out_path / filename.relative_to(path).parent
                out_subpath.mkdir(parents=True, exist_ok=True)
                out_filepath = out_subpath / filename.name
                click.echo("")
                self._process_file(filename, out_filepath, bar.pct)

        click.echo(f"Saved anonymized files to {out_path}")
        return out_path
