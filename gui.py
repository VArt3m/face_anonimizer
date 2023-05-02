import sys
from pathlib import Path
import traceback

from PySide6.QtWidgets import (
    QMainWindow,
    QApplication,
    QWidget,
    QPushButton,
    QFileDialog,
    QProgressBar,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QDoubleSpinBox,
    QFormLayout,
    QSpinBox,
    QLineEdit,
    QMessageBox,
)

from PySide6.QtCore import (
    QThread,
    Signal,
    Slot,
    Qt,
    QRunnable,
    QThreadPool,
    QObject,
    QSize,
)

from PySide6.QtGui import QIcon, Qt

from main import (
    FaceAnonymizer,
    supported_extensions,
    video_extensions,
    image_extensions,
)


class WorkerSignals(QObject):
    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)

    local_progress = Signal(float)
    global_progress = Signal(tuple)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # self.kwargs["local_progress_callback"] = self.signals.local_progress
        # self.kwargs["global_progress_callback"] = self.signals.global_progress

    @Slot()
    def run(self):
        """
        Initialise the runner function with passed args, kwargs.
        """

        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


class ParametersPanel(QGroupBox):
    def __init__(self):
        super().__init__()

        self.setTitle("Параметры")
        self.layout = QFormLayout()

        self.confidence_input = QDoubleSpinBox()
        self.confidence_input.setRange(0, 1)
        self.confidence_input.setSingleStep(0.1)
        self.confidence_input.setValue(0.5)
        self.confidence_input.setDecimals(1)
        self.confidence_input.setToolTip(
            "Порог уверенности для модели обнаружения лиц.\n"
            "Чем выше значение, тем увереннее модель должна быть для обнаружении лица.\n"
            "Принимает дробные значения от 0 до 1."
        )

        self.layout.addWidget(QLabel("Уверенность обнаружения лица"))
        self.layout.addWidget(self.confidence_input)

        self.face_min_size_input = QSpinBox()
        self.face_min_size_input.setRange(0, 100)
        self.face_min_size_input.setSingleStep(5)
        self.face_min_size_input.setValue(5)
        self.face_min_size_input.setSuffix("px")
        self.face_min_size_input.setToolTip(
            "Минимальный размер лица в пикселях.\n"
            "Принимает целые значения от 0 до 100."
        )

        self.layout.addWidget(QLabel("Минимальный размер лица"))
        self.layout.addWidget(self.face_min_size_input)

        self.face_expand_input = QDoubleSpinBox()
        self.face_expand_input.setRange(0, 1)
        self.face_expand_input.setSingleStep(0.1)
        self.face_expand_input.setValue(0.2)
        self.face_expand_input.setDecimals(1)
        self.face_expand_input.setToolTip(
            "Величина расширения области лица.\n"
            "Чем выше значение, тем больше будет расширена обнаруженная область лица для пикселизации.\n"
            "Принимает дробные значения от 0 до 1."
        )

        self.layout.addWidget(QLabel("Расширение области лица"))
        self.layout.addWidget(self.face_expand_input)

        self.pixelation_factor_input = QSpinBox()
        self.pixelation_factor_input.setRange(1, 150)
        self.pixelation_factor_input.setSingleStep(1)
        self.pixelation_factor_input.setValue(20)
        self.pixelation_factor_input.setToolTip(
            "Степень пикселизации лиц.\n"
            "Чем выше значение, тем сильнее пикселизируется лицо.\n"
            "Принимает целые значения от 1 до 150."
        )

        self.layout.addWidget(QLabel("Pixelation factor"))
        self.layout.addWidget(self.pixelation_factor_input)

        self.img_window_sizes = QLineEdit()
        self.img_window_sizes.setText("400, 700")
        self.img_window_sizes.setToolTip(
            "Размеры окон обнаружения лиц в пикселях.\n"
            "Чем больше и меньше (до разумного предела) окна, тем выше вероятность обнаружения лица.\n"
            "Но это также увеличивает время обработки.\n"
            "Принимает список целых чисел, разделенных запятыми."
        )

        self.layout.addWidget(QLabel("Размеры окон обнаружения лиц (не трогай)"))
        self.layout.addWidget(self.img_window_sizes)

        self.setLayout(self.layout)

    def get_parameters(self):
        return {
            "confidence": self.confidence_input.value(),
            "face_min_size": self.face_min_size_input.value(),
            "face_expand": self.face_expand_input.value(),
            "pixelation_factor": self.pixelation_factor_input.value(),
            "window_sizes": [int(x) for x in self.img_window_sizes.text().split(",")],
        }


class FilePanel(QGroupBox):
    def __init__(self):
        super().__init__()
        self.selected = None
        self.selected_type = None

        self.setTitle("Выбор файла или папки")

        self.layout = QGridLayout()

        self.select_file_button = QPushButton("Выбрать файл")
        self.select_file_button.clicked.connect(self.select_file)

        self.select_dir_button = QPushButton("Выбрать папку")
        self.select_dir_button.clicked.connect(self.select_dir)

        self.selected_label = QLabel("Файл или папка не выбран(а)")

        self.selected_display = QLineEdit()
        self.selected_display.setReadOnly(True)

        self.layout.addWidget(self.select_file_button, 0, 0)
        self.layout.addWidget(self.select_dir_button, 0, 1)
        self.layout.addWidget(self.selected_label, 1, 0, 1, 2)
        self.layout.addWidget(self.selected_display, 2, 0, 1, 2)

        self.setLayout(self.layout)

    def select_file(self):
        # Allow only images or videos
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл",
            "",
            "Изображения и Видео (*.png *.jpg *.jpeg *.mp4 *.avi *.mov);;Изображения (*.png *.jpg *.jpeg);;Видео (*.mp4 *.avi *.mov)",
        )

        if file_name:
            self.selected = Path(file_name)
            self.selected_type = (
                "image" if self.selected.suffix in image_extensions else "video"
            )

            self.selected_display.setText(str(self.selected))
            self.selected_label.setText(
                "Выбранное изображение"
                if self.selected_type == "image"
                else "Выбранное видео"
            )

    def select_dir(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Выберите папку")

        if dir_name:
            self.selected = Path(dir_name)
            self.selected_type = "dir"

            self.selected_display.setText(str(self.selected))
            self.selected_label.setText("Выбранная папка")


class ExecutionPanel(QGroupBox):
    def __init__(self, parent):
        super().__init__()
        self._parent = parent
        self.threadpool = QThreadPool()

        self.setTitle("Выполнение")

        self.layout = QVBoxLayout()

        self.start_button = QPushButton("Начать")
        self.start_button.clicked.connect(self.start)

        self.local_progress_label = QLabel("Прогресс обработки файла")

        self.local_progress = QProgressBar()
        self.local_progress.setValue(0)

        self.global_progress_label = QLabel("Прогресс обработки задачи")

        self.global_progress = QProgressBar()
        self.global_progress.setValue(0)

        self.layout.addWidget(self.start_button)
        self.layout.addWidget(self.local_progress_label)
        self.layout.addWidget(self.local_progress)
        self.layout.addWidget(self.global_progress_label)
        self.layout.addWidget(self.global_progress)

        self.setLayout(self.layout)

    def start(self):
        if self._parent.parameters_panel.get_parameters() is None:
            QMessageBox.critical(
                self,
                "Ошибка",
                "Параметры не выбраны или выбраны некорректно.\n"
                "Проверьте, что все параметры выбраны и введены корректно.",
            )
            return

        if self._parent.file_panel.selected is None:
            QMessageBox.critical(
                self,
                "Ошибка",
                "Файл или папка не выбраны.\n" "Проверьте, что файл или папка выбраны.",
            )
            return

        self.start_button.setEnabled(False)
        self.global_progress.setValue(0)
        self.local_progress.setValue(0)

        self.face_anonymizer = FaceAnonymizer(
            **self._parent.parameters_panel.get_parameters()
        )

        if self._parent.file_panel.selected_type == "dir":
            to_run = self.face_anonymizer.process_dir
        else:
            to_run = self.face_anonymizer.process_file

        self.worker = Worker(
            to_run,
            self._parent.file_panel.selected,
        )

        self.face_anonymizer.global_progress_callback = (
            self.worker.signals.global_progress.emit
        )
        self.face_anonymizer.local_progress_callback = (
            self.worker.signals.local_progress.emit
        )

        self.worker.signals.global_progress.connect(self.global_progress_update)
        self.worker.signals.local_progress.connect(self.local_progress_update)

        self.worker.signals.finished.connect(self.finish)
        self.worker.signals.error.connect(self.error)

        self.threadpool.start(self.worker)

    def global_progress_update(self, data):
        progress, filetype, filename = data
        self.local_progress.setValue(0 if filetype == "video" else 100)
        self.local_progress_label.setText(f"Обработка: {filename}")
        self.global_progress.setValue(int(progress * 100))

    def local_progress_update(self, progress):
        self.local_progress.setValue(int(progress * 100))

    def finish(self):
        self.start_button.setEnabled(True)
        self.local_progress_label.setText("Прогресс обработки файла")
        self.local_progress.setValue(0)
        self.global_progress.setValue(100)
        QMessageBox.information(self, "Успех", "Обработка завершена")

    def error(self, error):
        exctype, value, trcbck = error
        QMessageBox.critical(self, "Ошибка", str(value))
        self.start_button.setEnabled(True)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Анонимизатор лиц")

        self.layout = QHBoxLayout()
        sublayout = QVBoxLayout()

        self.parameters_panel = ParametersPanel()
        self.file_panel = FilePanel()
        self.execution_panel = ExecutionPanel(self)

        sublayout.addWidget(self.file_panel)
        sublayout.addStretch(1)
        sublayout.addWidget(self.execution_panel)

        self.layout.addWidget(self.parameters_panel)
        self.layout.addLayout(sublayout, stretch=1)

        self.widget = QWidget()
        self.widget.setLayout(self.layout)

        self.setCentralWidget(self.widget)


def set_taskbar_icon():
    import ctypes

    myappid = "artem30801.face_anonymizer.face_anonymizer"
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    app_icon = QIcon()
    icon_path = str(Path(__file__).parent / "icon.ico")
    app_icon.addFile(icon_path, QSize(256, 256))
    app.setWindowIcon(app_icon)

    if sys.platform == "win32":
        set_taskbar_icon()

    window = MainWindow()
    window.setWindowIcon(app_icon)
    window.showMaximized()

    app.exec()
