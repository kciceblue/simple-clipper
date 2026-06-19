from __future__ import annotations

from dataclasses import replace
import sys
import traceback
from pathlib import Path

from .media import FFmpegError, FFmpegRunner, build_clip_output_path, output_suffix_for_media
from .models import ClipSelection, MediaInfo, TranscriptSegment, TranscriptWord, format_timestamp
from .selection import TimedTranscriptItem, selections_from_indices
from .transcription import Transcriber, TranscriptionConfig


def main() -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("PySide6 is not installed. Run: python -m pip install -e .", file=sys.stderr)
        return 1

    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1120, 760)
    window.show()
    return app.exec()


def _qt_imports() -> dict[str, object]:
    from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QProgressBar,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    return locals()


qt = _qt_imports()
QObject = qt["QObject"]
QThread = qt["QThread"]
Signal = qt["Signal"]
Slot = qt["Slot"]
Qt = qt["Qt"]
QAbstractItemView = qt["QAbstractItemView"]
QCheckBox = qt["QCheckBox"]
QComboBox = qt["QComboBox"]
QFileDialog = qt["QFileDialog"]
QFormLayout = qt["QFormLayout"]
QGridLayout = qt["QGridLayout"]
QGroupBox = qt["QGroupBox"]
QHBoxLayout = qt["QHBoxLayout"]
QHeaderView = qt["QHeaderView"]
QLabel = qt["QLabel"]
QLineEdit = qt["QLineEdit"]
QMainWindow = qt["QMainWindow"]
QMessageBox = qt["QMessageBox"]
QProgressBar = qt["QProgressBar"]
QPushButton = qt["QPushButton"]
QTableWidget = qt["QTableWidget"]
QTableWidgetItem = qt["QTableWidgetItem"]
QTextEdit = qt["QTextEdit"]
QVBoxLayout = qt["QVBoxLayout"]
QWidget = qt["QWidget"]


class TranscriptionWorker(QObject):
    progress_changed = Signal(int, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, path: Path, config: TranscriptionConfig) -> None:
        super().__init__()
        self.path = path
        self.config = config

    @Slot()
    def run(self) -> None:
        try:
            result = Transcriber(self.config).transcribe(self.path, self.progress_changed.emit)
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(_format_exception(exc))


class ExportWorker(QObject):
    progress_changed = Signal(int, str)
    finished = Signal(list)
    failed = Signal(str)

    def __init__(
        self,
        source_path: Path,
        output_dir: Path,
        media_info: MediaInfo,
        clips: list[ClipSelection],
    ) -> None:
        super().__init__()
        self.source_path = source_path
        self.output_dir = output_dir
        self.media_info = media_info
        self.clips = clips

    @Slot()
    def run(self) -> None:
        try:
            runner = FFmpegRunner()
            suffix = output_suffix_for_media(self.media_info)
            exported: list[Path] = []
            total = len(self.clips)
            for offset, clip in enumerate(self.clips, start=1):
                output_path = build_clip_output_path(
                    self.source_path,
                    self.output_dir,
                    offset,
                    clip,
                    suffix,
                )
                self.progress_changed.emit(
                    int(((offset - 1) / total) * 100),
                    f"Exporting clip {offset} of {total}",
                )
                runner.export_clip(
                    input_path=self.source_path,
                    output_path=output_path,
                    selection=clip,
                    has_video=self.media_info.has_video,
                    video_stream_index=self.media_info.video_stream_index,
                )
                exported.append(output_path)
            self.progress_changed.emit(100, "Export complete")
            self.finished.emit(exported)
        except Exception as exc:
            self.failed.emit(_format_exception(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Simple Clipper")
        self.media_path: Path | None = None
        self.media_info: MediaInfo | None = None
        self.transcript_segments: tuple[TranscriptSegment, ...] = ()
        self.transcript_items: list[TimedTranscriptItem] = []
        self.clip_selections: list[ClipSelection] = []
        self.transcription_thread: QThread | None = None
        self.transcription_worker: TranscriptionWorker | None = None
        self.export_thread: QThread | None = None
        self.export_worker: ExportWorker | None = None
        self._updating_clips_table = False
        self._updating_clip_text_editor = False

        self._build_ui()
        self._set_busy(False)

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        source_group = QGroupBox("Source")
        source_layout = QGridLayout(source_group)
        self.open_button = QPushButton("Open media")
        self.open_button.clicked.connect(self.open_media)
        self.path_label = QLabel("No file selected")
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.media_label = QLabel("Install FFmpeg, then choose an audio or video file.")
        self.media_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        source_layout.addWidget(self.open_button, 0, 0)
        source_layout.addWidget(self.path_label, 0, 1)
        source_layout.addWidget(self.media_label, 1, 0, 1, 2)
        source_layout.setColumnStretch(1, 1)
        layout.addWidget(source_group)

        settings_group = QGroupBox("Transcription")
        settings_layout = QFormLayout(settings_group)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large-v3"])
        self.model_combo.setCurrentText("base")
        self.language_edit = QLineEdit()
        self.language_edit.setPlaceholderText("Auto detect")
        self.denoise_checkbox = QCheckBox("Voice-focused denoise")
        self.denoise_checkbox.setChecked(True)
        self.denoise_checkbox.setToolTip("Preprocess a temporary speech-focused WAV before transcription.")
        self.word_checkbox = QCheckBox("Enable word-level selection")
        self.word_checkbox.setToolTip("More precise text clipping, with extra processing time.")
        self.transcribe_button = QPushButton("Transcribe")
        self.transcribe_button.clicked.connect(self.start_transcription)
        settings_layout.addRow("Model", self.model_combo)
        settings_layout.addRow("Language", self.language_edit)
        settings_layout.addRow("", self.denoise_checkbox)
        settings_layout.addRow("", self.word_checkbox)
        settings_layout.addRow("", self.transcribe_button)
        layout.addWidget(settings_group)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status_label = QLabel("Ready")
        layout.addWidget(self.progress)
        layout.addWidget(self.status_label)

        self.transcript_table = QTableWidget(0, 3)
        self.transcript_table.setHorizontalHeaderLabels(["Start", "End", "Text"])
        self.transcript_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.transcript_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.transcript_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.transcript_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.transcript_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.transcript_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        layout.addWidget(self.transcript_table, stretch=3)

        clip_buttons = QHBoxLayout()
        self.add_clip_button = QPushButton("Add selected text as clip")
        self.add_clip_button.clicked.connect(self.add_selected_clips)
        self.remove_clip_button = QPushButton("Remove selected clip")
        self.remove_clip_button.clicked.connect(self.remove_selected_clips)
        self.copy_clip_text_button = QPushButton("Copy selected text")
        self.copy_clip_text_button.clicked.connect(self.copy_selected_clip_text)
        self.export_button = QPushButton("Export clips")
        self.export_button.clicked.connect(self.export_clips)
        clip_buttons.addWidget(self.add_clip_button)
        clip_buttons.addWidget(self.remove_clip_button)
        clip_buttons.addWidget(self.copy_clip_text_button)
        clip_buttons.addStretch(1)
        clip_buttons.addWidget(self.export_button)
        layout.addLayout(clip_buttons)

        self.clips_table = QTableWidget(0, 4)
        self.clips_table.setHorizontalHeaderLabels(["Start", "End", "Duration", "Text"])
        self.clips_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.clips_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.clips_table.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed | QAbstractItemView.AnyKeyPressed
        )
        self.clips_table.itemChanged.connect(self._on_clip_table_item_changed)
        self.clips_table.itemSelectionChanged.connect(self._on_clip_selection_changed)
        self.clips_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.clips_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.clips_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.clips_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        layout.addWidget(self.clips_table, stretch=1)

        self.clip_text_editor = QTextEdit()
        self.clip_text_editor.setPlaceholderText("Select one merged clip to edit its text.")
        self.clip_text_editor.setMaximumHeight(120)
        self.clip_text_editor.setReadOnly(True)
        self.clip_text_editor.textChanged.connect(self._on_clip_text_editor_changed)
        layout.addWidget(self.clip_text_editor)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(100)
        layout.addWidget(self.log_box)

        self.setCentralWidget(root)
        self._refresh_action_state()

    @Slot()
    def open_media(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open audio or video",
            str(Path.home()),
            "Media files (*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.mp4 *.mov *.mkv *.webm *.avi);;All files (*)",
        )
        if not path:
            return

        media_path = Path(path)
        try:
            info = FFmpegRunner().probe(media_path)
        except FFmpegError as exc:
            self._show_error("Media error", str(exc))
            return

        self.media_path = media_path
        self.media_info = info
        self.transcript_segments = ()
        self.transcript_items = []
        self.clip_selections = []
        self.path_label.setText(str(media_path))
        self.media_label.setText(
            f"{info.media_kind.title()} | duration {format_timestamp(info.duration)} | format {info.format_name or 'unknown'}"
        )
        self._populate_transcript_table()
        self._populate_clips_table()
        self._set_status("Media loaded")
        self._refresh_action_state()

    @Slot()
    def start_transcription(self) -> None:
        if self.media_path is None:
            self._show_error("No media", "Choose an audio or video file first.")
            return

        language = self.language_edit.text().strip() or None
        config = TranscriptionConfig(
            model_size=self.model_combo.currentText(),
            word_timestamps=self.word_checkbox.isChecked(),
            language=language,
            voice_denoise=self.denoise_checkbox.isChecked(),
        )
        self._set_busy(True)
        self.progress.setValue(0)
        self.clip_selections = []
        self._populate_clips_table()
        self._set_status("Preparing transcription")

        self.transcription_thread = QThread(self)
        self.transcription_worker = TranscriptionWorker(self.media_path, config)
        self.transcription_worker.moveToThread(self.transcription_thread)
        self.transcription_thread.started.connect(self.transcription_worker.run)
        self.transcription_worker.progress_changed.connect(self._on_progress)
        self.transcription_worker.finished.connect(self._on_transcription_finished)
        self.transcription_worker.failed.connect(self._on_worker_failed)
        self.transcription_worker.finished.connect(self.transcription_thread.quit)
        self.transcription_worker.failed.connect(self.transcription_thread.quit)
        self.transcription_thread.finished.connect(self._clear_transcription_worker)
        self.transcription_thread.start()

    @Slot(object)
    def _on_transcription_finished(self, result: object) -> None:
        self.transcript_segments = result.segments
        self.transcript_items = self._flatten_transcript_items(result.word_timestamps)
        self._populate_transcript_table()
        language = result.language or "unknown"
        self._log(f"Transcription complete. Language: {language}. Segments: {len(result.segments)}.")
        self._set_status("Transcription complete")
        self._set_busy(False)

    @Slot()
    def add_selected_clips(self) -> None:
        rows = sorted({index.row() for index in self.transcript_table.selectionModel().selectedRows()})
        if not rows:
            self._show_error("No transcript selection", "Select one or more transcript rows first.")
            return

        try:
            new_clips = selections_from_indices(self.transcript_items, rows)
        except IndexError:
            self._show_error("Invalid selection", "The selected transcript rows are no longer available.")
            return

        self.clip_selections.extend(new_clips)
        self._populate_clips_table()
        self._set_status(f"Added {len(new_clips)} clip selection(s)")
        self._refresh_action_state()

    @Slot()
    def remove_selected_clips(self) -> None:
        rows = sorted({index.row() for index in self.clips_table.selectionModel().selectedRows()}, reverse=True)
        for row in rows:
            if 0 <= row < len(self.clip_selections):
                del self.clip_selections[row]
        self._populate_clips_table()
        self._refresh_clip_text_editor()
        self._refresh_action_state()

    @Slot()
    def copy_selected_clip_text(self) -> None:
        text = self._selected_clip_text()
        if not text:
            self._show_error("No clip text", "Select at least one merged clip first.")
            return
        self.clip_text_editor.copy()
        if not self.clip_text_editor.textCursor().hasSelection():
            self.clip_text_editor.selectAll()
            self.clip_text_editor.copy()
        self.clip_text_editor.setFocus()
        self._set_status("Copied selected clip text")

    @Slot()
    def export_clips(self) -> None:
        if self.media_path is None or self.media_info is None:
            self._show_error("No media", "Choose an audio or video file first.")
            return
        if not self.clip_selections:
            self._show_error("No clips", "Add at least one clip selection first.")
            return
        self._sync_clip_text_editor_to_selection()

        output_dir = QFileDialog.getExistingDirectory(self, "Choose export folder", str(self.media_path.parent))
        if not output_dir:
            return

        self._set_busy(True)
        self.progress.setValue(0)
        self.export_thread = QThread(self)
        self.export_worker = ExportWorker(
            source_path=self.media_path,
            output_dir=Path(output_dir),
            media_info=self.media_info,
            clips=list(self.clip_selections),
        )
        self.export_worker.moveToThread(self.export_thread)
        self.export_thread.started.connect(self.export_worker.run)
        self.export_worker.progress_changed.connect(self._on_progress)
        self.export_worker.finished.connect(self._on_export_finished)
        self.export_worker.failed.connect(self._on_worker_failed)
        self.export_worker.finished.connect(self.export_thread.quit)
        self.export_worker.failed.connect(self.export_thread.quit)
        self.export_thread.finished.connect(self._clear_export_worker)
        self.export_thread.start()

    @Slot(list)
    def _on_export_finished(self, exported: list[Path]) -> None:
        for path in exported:
            self._log(f"Exported {path}")
        self._set_status(f"Exported {len(exported)} clip(s)")
        self._set_busy(False)

    @Slot(int, str)
    def _on_progress(self, value: int, message: str) -> None:
        self.progress.setValue(value)
        self._set_status(message)

    @Slot(str)
    def _on_worker_failed(self, message: str) -> None:
        self._set_busy(False)
        self._show_error("Operation failed", message)
        self._log(message)

    @Slot()
    def _clear_transcription_worker(self) -> None:
        self.transcription_worker = None
        self.transcription_thread = None
        self._refresh_action_state()

    @Slot()
    def _clear_export_worker(self) -> None:
        self.export_worker = None
        self.export_thread = None
        self._refresh_action_state()

    def _flatten_transcript_items(self, word_timestamps: bool) -> list[TimedTranscriptItem]:
        if not word_timestamps:
            return list(self.transcript_segments)

        words: list[TranscriptWord] = []
        for segment in self.transcript_segments:
            words.extend(segment.words)
        return words or list(self.transcript_segments)

    def _populate_transcript_table(self) -> None:
        self.transcript_table.setRowCount(len(self.transcript_items))
        for row, item in enumerate(self.transcript_items):
            self._set_table_item(self.transcript_table, row, 0, format_timestamp(item.start))
            self._set_table_item(self.transcript_table, row, 1, format_timestamp(item.end))
            self._set_table_item(self.transcript_table, row, 2, item.text.strip())
        self._refresh_action_state()

    def _populate_clips_table(self) -> None:
        self._updating_clips_table = True
        self.clips_table.setRowCount(len(self.clip_selections))
        for row, clip in enumerate(self.clip_selections):
            self._set_table_item(self.clips_table, row, 0, format_timestamp(clip.start), editable=False)
            self._set_table_item(self.clips_table, row, 1, format_timestamp(clip.end), editable=False)
            self._set_table_item(self.clips_table, row, 2, f"{clip.duration:.3f}s", editable=False)
            self._set_table_item(self.clips_table, row, 3, clip.text, editable=True)
        self._updating_clips_table = False
        self._refresh_clip_text_editor()
        self._refresh_action_state()

    def _set_busy(self, busy: bool) -> None:
        for widget in (
            self.open_button,
            self.transcribe_button,
            self.model_combo,
            self.language_edit,
            self.denoise_checkbox,
            self.word_checkbox,
            self.add_clip_button,
            self.remove_clip_button,
            self.copy_clip_text_button,
            self.clip_text_editor,
            self.export_button,
        ):
            widget.setEnabled(not busy)
        if not busy:
            self._refresh_action_state()

    def _refresh_action_state(self) -> None:
        has_media = self.media_path is not None
        has_transcript = bool(self.transcript_items)
        has_clips = bool(self.clip_selections)
        self.transcribe_button.setEnabled(has_media)
        self.add_clip_button.setEnabled(has_transcript)
        self.remove_clip_button.setEnabled(has_clips)
        self.copy_clip_text_button.setEnabled(has_clips)
        self.export_button.setEnabled(has_media and has_clips)

    @Slot()
    def _on_clip_selection_changed(self) -> None:
        self._refresh_clip_text_editor()

    @Slot(object)
    def _on_clip_table_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_clips_table or item.column() != 3:
            return
        row = item.row()
        if not 0 <= row < len(self.clip_selections):
            return
        self._update_clip_text(row, item.text())
        selected_rows = self._selected_clip_rows()
        if selected_rows == [row]:
            self._set_clip_text_editor_text(item.text())

    @Slot()
    def _on_clip_text_editor_changed(self) -> None:
        if self._updating_clip_text_editor:
            return
        selected_rows = self._selected_clip_rows()
        if len(selected_rows) != 1:
            return
        row = selected_rows[0]
        text = self.clip_text_editor.toPlainText()
        self._update_clip_text(row, text)
        table_item = self.clips_table.item(row, 3)
        if table_item is not None and table_item.text() != text:
            self._updating_clips_table = True
            table_item.setText(text)
            self._updating_clips_table = False

    def _refresh_clip_text_editor(self) -> None:
        selected_rows = self._selected_clip_rows()
        if len(selected_rows) == 1:
            row = selected_rows[0]
            self._set_clip_text_editor_text(self.clip_selections[row].text)
            self.clip_text_editor.setReadOnly(False)
            self.clip_text_editor.setPlaceholderText("Edit the merged clip text before export.")
        elif len(selected_rows) > 1:
            self._set_clip_text_editor_text(self._selected_clip_text())
            self.clip_text_editor.setReadOnly(True)
            self.clip_text_editor.setPlaceholderText("Multiple clips selected. Copy is available; edit one clip at a time.")
        else:
            self._set_clip_text_editor_text("")
            self.clip_text_editor.setReadOnly(True)
            self.clip_text_editor.setPlaceholderText("Select one merged clip to edit its text.")

    def _set_clip_text_editor_text(self, text: str) -> None:
        if self.clip_text_editor.toPlainText() == text:
            return
        self._updating_clip_text_editor = True
        self.clip_text_editor.setPlainText(text)
        self._updating_clip_text_editor = False

    def _sync_clip_text_editor_to_selection(self) -> None:
        selected_rows = self._selected_clip_rows()
        if len(selected_rows) == 1 and not self.clip_text_editor.isReadOnly():
            self._update_clip_text(selected_rows[0], self.clip_text_editor.toPlainText())

    def _selected_clip_rows(self) -> list[int]:
        return sorted({index.row() for index in self.clips_table.selectionModel().selectedRows()})

    def _selected_clip_text(self) -> str:
        rows = self._selected_clip_rows()
        if not rows:
            return ""
        return "\n\n".join(self.clip_selections[row].text for row in rows if 0 <= row < len(self.clip_selections))

    def _update_clip_text(self, row: int, text: str) -> None:
        if 0 <= row < len(self.clip_selections):
            self.clip_selections[row] = replace(self.clip_selections[row], text=text)

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _log(self, message: str) -> None:
        self.log_box.append(message)

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)

    @staticmethod
    def _set_table_item(table: QTableWidget, row: int, column: int, value: str, editable: bool = False) -> None:
        item = QTableWidgetItem(value)
        if editable:
            item.setFlags(item.flags() | Qt.ItemIsEditable)
        else:
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        table.setItem(row, column, item)


def _format_exception(exc: Exception) -> str:
    details = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    return details or str(exc)
