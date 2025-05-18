import sys
import asyncio # Import asyncio
import logging # Import logging
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QTextBrowser, QLabel, QProgressBar, QStatusBar,
    QFileDialog, QSizePolicy, QSplitter, QListWidget, QListWidgetItem, QFrame
)
from PyQt5.QtCore import QObject, pyqtSignal, QRunnable, QThreadPool, Qt, QSize, pyqtSlot # Import pyqtSlot
from PyQt5.QtGui import QColor, QTextCursor, QSyntaxHighlighter, QTextCharFormat, QFont, QIcon # Add QIcon
from PyQt5.QtCore import QRegularExpression # Import QRegularExpression here

# Assuming main logic is refactored as planned
from main import run_agent_task

# --- Constants ---
CONFIG_PATH = Path("config.yaml")
CONTEXT_PATH = Path("context.json")
MAX_LINES_TO_DISPLAY = 1000 # Limit for large files

# --- Logging Setup ---
class GUILogHandler(logging.Handler, QObject):
    """Custom logging handler that emits a PyQt signal."""
    log_signal = pyqtSignal(str, str) # message, level_name

    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self) # Initialize QObject part

    def emit(self, record):
        """Emit a signal with the formatted log message and level name."""
        log_entry = self.format(record)
        self.log_signal.emit(log_entry, record.levelname)

# --- Worker Signals and Runnable ---
class WorkerSignals(QObject):
    '''Defines signals available from a running worker thread.'''
    log_message = pyqtSignal(str, str)  # message, level (e.g., "info", "error") - Kept for potential direct use, but logging framework is preferred
    task_complete = pyqtSignal(str)     # final result string
    progress_update = pyqtSignal(int)   # Optional: for finer progress
    status_update = pyqtSignal(str)     # e.g., "Running...", "Idle"
    file_focus_update = pyqtSignal(str) # Signal with the path of the file being processed
    file_content_update = pyqtSignal(str, str) # Signal with file path and its updated content

class AgentWorker(QRunnable):
    '''Worker object (QRunnable) to run the agent task in the thread pool.'''

    def __init__(self, task: str, working_dir: Path | None, signals: WorkerSignals):
        super().__init__()
        self.task = task
        self.working_dir = working_dir # Store the working directory
        self.signals = signals # Use the passed signals object
        self.is_running = True
        self.setAutoDelete(True) # Auto-delete when done

    def run(self):
        '''Execute the agent task.'''
        try:
            # Use constants defined at the top
            config_path = CONFIG_PATH
            context_path = CONTEXT_PATH

            # --- Logging Callback using Python's logging ---
            # The GUILogHandler will route these to the GUI
            def log_callback(message: str, level: str = "info", **kwargs):
                if self.is_running:
                    level_upper = level.upper()
                    if level_upper == "INFO":
                        logging.info(message)
                    elif level_upper == "WARNING":
                        logging.warning(message)
                    elif level_upper == "ERROR":
                        logging.error(message)
                    elif level_upper == "DEBUG":
                        logging.debug(message)
                    elif level_upper == "SUCCESS": # Custom level handling
                        logging.log(logging.INFO + 5, message) # Use a custom level if needed or map to INFO
                    else:
                        logging.info(message) # Default to info

            # --- Placeholder for File Focus Callback ---
            def qt_file_focus_callback(file_path: str):
                 if self.is_running:
                     # Construct absolute path if needed, assuming file_path is relative to working_dir
                     abs_path = self.working_dir / file_path if self.working_dir else Path(file_path)
                     self.signals.file_focus_update.emit(str(abs_path))


            # --- Callback for Real-time Content Updates ---
            def qt_file_content_callback(file_path: str, new_content: str):
                if self.is_running:
                    # Construct absolute path if needed
                    abs_path = self.working_dir / file_path if self.working_dir else Path(file_path)
                    self.signals.file_content_update.emit(str(abs_path), new_content)

            self.signals.status_update.emit(f"Running in {self.working_dir}...") # Update status
            # run_agent_task is now async, so we need asyncio.run()
            final_result = asyncio.run(run_agent_task(
                task=self.task,
                config_path=config_path,
                context_path=context_path if context_path.exists() else None,
                log_callback=log_callback, # Use the new logging callback
                working_directory=self.working_dir, # Pass it along
                file_focus_callback=qt_file_focus_callback, # Pass the focus callback
                file_content_callback=qt_file_content_callback # Pass the content update callback
            ))
            if self.is_running:
                self.signals.task_complete.emit(str(final_result))

        except Exception as e:
            if self.is_running:
                logging.error(f"An unexpected error occurred in worker: {e}", exc_info=True) # Log error with traceback
                self.signals.task_complete.emit(f"Error: {e}") # Signal completion even on error
        finally:
            if self.is_running:
                self.signals.status_update.emit("Idle")

    def stop(self):
        # This provides a way to signal the worker to stop,
        # but the worker's loop/task needs to check self.is_running periodically.
        logging.info("Stop signal received by worker.")
        self.is_running = False

# --- Syntax Highlighter ---
# TODO: Optimize highlighter for large files. Consider:
# - Applying rules only to visible parts of the document.
# - Using more efficient regex patterns or state machines.
# - Exploring external libraries like Pygments if performance is critical.
class PythonHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlightingRules = []

        # Keyword format
        keywordFormat = QTextCharFormat()
        keywordFormat.setForeground(QColor("#569CD6")) # Light Blue
        keywordFormat.setFontWeight(QFont.Bold)
        keywords = [
            "\\bFalse\\b", "\\bNone\\b", "\\bTrue\\b", "\\bas\\b", "\\bassert\\b",
            "\\basync\\b", "\\bawait\\b", "\\bbreak\\b", "\\bclass\\b", "\\bcontinue\\b",
            "\\bdef\\b", "\\bdel\\b", "\\belif\\b", "\\belse\\b", "\\bexcept\\b",
            "\\bfinally\\b", "\\bfor\\b", "\\bfrom\\b", "\\bglobal\\b", "\\bif\\b",
            "\\bimport\\b", "\\bin\\b", "\\bis\\b", "\\blambda\\b", "\\bnonlocal\\b",
            "\\bnot\\b", "\\bor\\b", "\\bpass\\b", "\\braise\\b", "\\breturn\\b",
            "\\btry\\b", "\\bwhile\\b", "\\bwith\\b", "\\byield\\b"
        ]
        for word in keywords:
            pattern = QRegularExpression(word)
            rule = (pattern, keywordFormat)
            self.highlightingRules.append(rule)

        # Built-in functions/types format (optional, can expand)
        builtinFormat = QTextCharFormat()
        builtinFormat.setForeground(QColor("#4EC9B0")) # Teal
        builtins = [
            "\\bprint\\b", "\\blen\\b", "\\bstr\\b", "\\bint\\b", "\\bfloat\\b",
            "\\blist\\b", "\\bdict\\b", "\\bset\\b", "\\btuple\\b", "\\btype\\b",
            "\\bsuper\\b", "\\brange\\b", "\\benumerate\\b", "\\bzip\\b",
            "\\bException\\b", "\\bself\\b" # Add self
        ]
        for word in builtins:
             pattern = QRegularExpression(word)
             rule = (pattern, builtinFormat)
             self.highlightingRules.append(rule)

        # Class format
        classFormat = QTextCharFormat()
        classFormat.setFontWeight(QFont.Bold)
        classFormat.setForeground(QColor("#4EC9B0")) # Teal
        self.highlightingRules.append((QRegularExpression("\\bclass\\s+[A-Za-z_][A-Za-z0-9_]*"), classFormat))

        # Function format
        functionFormat = QTextCharFormat()
        # functionFormat.setFontItalic(True) # Optional italic
        functionFormat.setForeground(QColor("#DCDCAA")) # Yellowish
        self.highlightingRules.append((QRegularExpression("\\bdef\\s+[A-Za-z_][A-Za-z0-9_]*"), functionFormat))

        # Single-line comment format
        singleLineCommentFormat = QTextCharFormat()
        singleLineCommentFormat.setForeground(QColor("#6A9955")) # Green
        singleLineCommentFormat.setFontItalic(True)
        self.highlightingRules.append((QRegularExpression("#[^\n]*"), singleLineCommentFormat))

        # Quotation format (strings)
        quotationFormat = QTextCharFormat()
        quotationFormat.setForeground(QColor("#CE9178")) # Orange/Brown
        # Use QRegularExpression for strings, handle escapes better potentially
        # Basic patterns, might need refinement for edge cases like escaped quotes within strings
        self.highlightingRules.append((QRegularExpression("\".*?(?<!\\\\)\""), quotationFormat)) # Double quotes
        self.highlightingRules.append((QRegularExpression("'.*?(?<!\\\\)'"), quotationFormat)) # Single quotes
        # Basic multi-line string formats (won't handle complex cases perfectly)
        self.multiLineStringFormat = QTextCharFormat()
        self.multiLineStringFormat.setForeground(QColor("#CE9178")) # Orange/Brown
        self.tripleSingleQuoteStart = QRegularExpression("'''")
        self.tripleDoubleQuoteStart = QRegularExpression('"""')


    def highlightBlock(self, text):
        # Apply single-line rules using QRegularExpressionIterator
        for pattern, format in self.highlightingRules:
            iterator = pattern.globalMatch(text)
            while iterator.hasNext():
                match = iterator.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), format)

        self.setCurrentBlockState(0) # Default state

        # Handle multi-line strings (basic implementation)
        in_multiline_string = (self.previousBlockState() == 1)
        startIndex = 0

        while startIndex < len(text):
            if not in_multiline_string:
                # Check for start of multi-line string using QRegularExpression
                single_start_match = self.tripleSingleQuoteStart.match(text, startIndex)
                double_start_match = self.tripleDoubleQuoteStart.match(text, startIndex)

                single_start_index = single_start_match.capturedStart() if single_start_match.hasMatch() else -1
                double_start_index = double_start_match.capturedStart() if double_start_match.hasMatch() else -1

                # Find the earliest start index
                start_index_found = -1
                start_pattern = None
                end_pattern = None

                if single_start_index != -1 and (double_start_index == -1 or single_start_index < double_start_index):
                    start_index_found = single_start_index
                    start_pattern = self.tripleSingleQuoteStart
                    end_pattern = self.tripleSingleQuoteStart # Corresponding end pattern
                elif double_start_index != -1:
                    start_index_found = double_start_index
                    start_pattern = self.tripleDoubleQuoteStart
                    end_pattern = self.tripleDoubleQuoteStart # Corresponding end pattern

                if start_index_found != -1:
                    in_multiline_string = True
                    startIndex = start_index_found # Move to the start of the string
                else:
                    break # No more multi-line starts in the rest of the block

            if in_multiline_string:
                # Determine which end pattern corresponds to the start
                if self.previousBlockState() == 1: # If continuing from previous block, need to know which quote type it was
                   # If continuing from a previous block, we don't know which quote started it
                   # without storing more state. Let's check for both end patterns.
                   # A more robust highlighter would store the type of quote in the block state.
                   single_end_match = self.tripleSingleQuoteStart.match(text, startIndex)
                   double_end_match = self.tripleDoubleQuoteStart.match(text, startIndex)
                   single_end_index = single_end_match.capturedStart() if single_end_match.hasMatch() else -1
                   double_end_index = double_end_match.capturedStart() if double_end_match.hasMatch() else -1

                   end_index = -1
                   end_pattern_found = None
                   if single_end_index != -1 and (double_end_index == -1 or single_end_index < double_end_index):
                       end_index = single_end_index
                       end_pattern_found = self.tripleSingleQuoteStart
                   elif double_end_index != -1:
                       end_index = double_end_index
                       end_pattern_found = self.tripleDoubleQuoteStart

                else: # Started in this block, we know the end pattern
                    end_match = end_pattern.match(text, startIndex + 3) # Search *after* the opening quotes
                    end_index = end_match.capturedStart() if end_match.hasMatch() else -1
                    end_pattern_found = end_pattern # Keep track of the pattern found


                if end_index == -1: # String continues to the next block
                    self.setCurrentBlockState(1)
                    length = len(text) - startIndex
                    self.setFormat(startIndex, length, self.multiLineStringFormat)
                    break # End of block reached
                else: # String ends in this block
                    length = end_index - startIndex + 3 # Closing quotes are always 3 chars
                    self.setFormat(startIndex, length, self.multiLineStringFormat)
                    in_multiline_string = False
                    startIndex += length # Move past the end of this string
            else: # Should not be reached if logic is correct
                 break

# --- File Viewer Widget (Right Panel) ---
class FileViewerWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("rightFrame")
        self.setFrameShape(QFrame.NoFrame) # No frame initially, styled by CSS
        self._current_file_path = None # Store the full path of the currently displayed file

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10) # Padding inside right panel
        layout.setSpacing(8)

        # Top Info (File Path / Status)
        self.file_progress_label = QLabel("MaS's Computer: Idle") # Changed label
        # Add Diff/Original/Modified buttons (placeholders)
        file_view_options_layout = QHBoxLayout()
        file_view_options_layout.addStretch()
        self.diff_button = QPushButton("Diff")
        self.original_button = QPushButton("Original")
        self.modified_button = QPushButton("Modified")
        self.diff_button.setEnabled(False) # Disable placeholders for now
        self.original_button.setEnabled(False)
        self.modified_button.setEnabled(False)
        file_view_options_layout.addWidget(self.diff_button)
        file_view_options_layout.addWidget(self.original_button)
        file_view_options_layout.addWidget(self.modified_button)


        # File Content Display
        self.file_progress_display = QTextBrowser()
        self.file_progress_display.setObjectName("fileBrowser") # For specific styling
        self.file_progress_display.setReadOnly(True) # Or False if editing is intended later
        self.highlighter = PythonHighlighter(self.file_progress_display.document()) # Apply highlighter

        # Add widgets to layout
        layout.addWidget(self.file_progress_label)
        layout.addLayout(file_view_options_layout)
        layout.addWidget(self.file_progress_display, 1) # Display takes available space

    def clear_display(self):
        """Clears the file display and resets the label."""
        self.file_progress_display.clear()
        self.file_progress_label.setText("MaS's Computer: Idle")
        self._current_file_path = None

    @pyqtSlot(str) # Explicitly mark as a slot
    def set_file_to_display(self, file_path_str: str):
        """Reads the specified file and displays its content (or partial content)."""
        self._current_file_path = Path(file_path_str) # Store the full path
        file_name = self._current_file_path.name
        self.file_progress_label.setText(f"MaS's Computer: Viewing {file_name}") # Update label
        content_to_display = ""
        try:
            file_path = self._current_file_path
            if not file_path.exists():
                content_to_display = f"[File not found or not yet created: {file_path_str}]"
                logging.warning(f"File not found or not yet created: {file_path_str}")
            elif not file_path.is_file():
                content_to_display = f"[Path exists but is not a file: {file_path_str}]"
                logging.warning(f"Path exists but is not a file: {file_path_str}")
            else:
                # Read file content with specific error handling and size limit
                try:
                    lines = []
                    line_count = 0
                    is_truncated = False
                    with file_path.open('r', encoding='utf-8') as f:
                        for line in f:
                            if line_count < MAX_LINES_TO_DISPLAY:
                                lines.append(line)
                            else:
                                is_truncated = True
                                break # Stop reading after max lines
                            line_count += 1

                    content_to_display = "".join(lines)
                    if is_truncated:
                        content_to_display += f"\n\n[--- File truncated: Displaying first {MAX_LINES_TO_DISPLAY} lines ---]"
                        logging.info(f"Displayed truncated content (first {MAX_LINES_TO_DISPLAY} lines) for large file: {file_path_str}")

                    # Note: Highlighting filename in task input is moved out as it depends on task_input widget

                except FileNotFoundError: # Should be caught by exists() check, but good practice
                     content_to_display = f"[Error: File not found: {file_path_str}]"
                     logging.error(f"File not found during read attempt: {file_path_str}")
                except PermissionError:
                    content_to_display = f"[Error: Permission denied to read file: {file_name}]"
                    logging.error(f"Permission denied reading file: {file_path_str}")
                except UnicodeDecodeError:
                    content_to_display = f"[Error: Could not decode file '{file_name}' as UTF-8. It might be binary or use a different encoding.]"
                    logging.warning(f"Could not decode file '{file_name}' as UTF-8.")
                except OSError as os_err: # Catch other potential OS errors during file read
                    content_to_display = f"[OS Error reading file: {os_err}]"
                    logging.error(f"OS Error reading file '{file_path_str}': {os_err}", exc_info=True)
                except Exception as read_err: # Catch-all for unexpected errors during read
                    content_to_display = f"[Unexpected error reading file: {read_err}]"
                    logging.error(f"Unexpected error reading file '{file_path_str}': {read_err}", exc_info=True)

        except Exception as path_err: # Catch errors related to Path object itself
            logging.error(f"Error processing file path '{file_path_str}': {path_err}", exc_info=True)
            content_to_display = f"[Error processing path: {path_err}]"
            self._current_file_path = None # Reset path if processing failed

        # Update the display and scroll to top
        self.file_progress_display.setPlainText(content_to_display)
        self.file_progress_display.moveCursor(QTextCursor.Start)

    @pyqtSlot(str, str) # Explicitly mark as a slot
    def update_displayed_content(self, file_path_str: str, new_content: str):
        """Updates the file display only if the path matches the currently displayed file."""
        # Check if the update corresponds to the file currently being displayed
        if self._current_file_path and self._current_file_path == Path(file_path_str):
            # Check if content is truncated (basic check for the truncation message)
            is_truncated = f"[--- File truncated: Displaying first {MAX_LINES_TO_DISPLAY} lines ---]" in self.file_progress_display.toPlainText()

            display_content = new_content
            # If the original view was truncated, only show the beginning of the new content
            if is_truncated:
                 lines = new_content.splitlines(True) # Keep line endings
                 if len(lines) > MAX_LINES_TO_DISPLAY:
                     display_content = "".join(lines[:MAX_LINES_TO_DISPLAY])
                     display_content += f"\n\n[--- File truncated: Displaying first {MAX_LINES_TO_DISPLAY} lines ---]"
                     logging.info(f"Displayed updated truncated content for {file_path_str}")
                 # else: display_content remains the full new_content if it's now shorter

            self.file_progress_display.setPlainText(display_content)
            self.file_progress_display.moveCursor(QTextCursor.Start) # Scroll to top
            logging.debug(f"Updated displayed content for focused file: {file_path_str}")
        else:
            logging.debug(f"Content update for non-focused file ignored: {file_path_str}")


# --- Main GUI Window ---
class AgentAppGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MaS - Multi-Agent System") # Changed title
        self.setGeometry(50, 50, 1200, 700) # Adjusted size for 3 columns
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F3F3F3; /* Light gray background */
            }
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif; /* Consistent font */
                font-size: 9pt;
            }
            QTextBrowser, QTextEdit {
                border: 1px solid #D1D1D1;
                border-radius: 4px;
                background-color: white;
                color: #333;
            }
            QPushButton {
                background-color: #E1E1E1;
                border: 1px solid #CCCCCC;
                padding: 5px 10px;
                border-radius: 4px;
                min-height: 20px; /* Ensure buttons have some height */
            }
            QPushButton:hover {
                background-color: #D1D1D1;
                border: 1px solid #BDBDBD;
            }
            QPushButton:pressed {
                background-color: #C1C1C1;
            }
            QLabel {
                color: #555;
                padding-bottom: 3px; /* Add spacing below labels */
            }
            QStatusBar {
                background-color: #EAEAEA;
            }
            QStatusBar QLabel {
                padding-bottom: 0px; /* Reset padding for status bar labels */
            }
            QListWidget {
                border: none; /* Remove border from list widget itself */
                background-color: #EAEAEA; /* Slightly darker sidebar */
            }
            QListWidget::item {
                padding: 8px 10px;
                border-bottom: 1px solid #DCDCDC;
            }
            QListWidget::item:selected {
                background-color: #D0E0F0; /* Light blue selection */
                color: #111;
            }
            QFrame#middleFrame, QFrame#rightFrame { /* Style frames for visual separation */
                 border-left: 1px solid #D1D1D1;
            }
            QTextBrowser#logBrowser { /* Specific styling for log */
                 background-color: #FFFFFF;
                 color: #333333;
            }
             QTextBrowser#fileBrowser { /* Specific styling for file view */
                 background-color: #FFFFFF;
                 color: #333333;
                 font-family: Consolas, monospace; /* Monospaced font */
            }
            QPushButton#newTaskButton { /* Specific style for New Task */
                background-color: #FFFFFF;
                text-align: left;
                padding-left: 10px;
                font-weight: bold;
                border: 1px solid #D1D1D1;
            }
            QPushButton#newTaskButton:hover {
                background-color: #F0F0F0;
            }
        """)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget) # Main layout is horizontal
        self.main_layout.setContentsMargins(0, 0, 0, 0) # No margins for the main layout
        self.main_layout.setSpacing(0) # No spacing between columns

        self.working_directory = None # Initialize working directory path
        self.worker_signals = WorkerSignals() # Instantiate signals here
        # self.setup_logging() # Setup logging framework - MOVED TO END OF __init__

        # --- Create Splitter for Resizable Columns ---
        self.splitter = QSplitter(Qt.Horizontal)

        # --- Add Sidebar Toggle Button ---
        self.sidebar_toggle_button = QPushButton("<") # Initial state: sidebar visible
        self.sidebar_toggle_button.setFixedWidth(25) # Small fixed width
        self.sidebar_toggle_button.setToolTip("Toggle Sidebar")
        self.sidebar_toggle_button.clicked.connect(self.toggle_sidebar)
        self.sidebar_toggle_button.setStyleSheet("QPushButton { min-height: 0; padding: 4px; }") # Compact style

        # Add toggle button and splitter to main layout
        self.main_layout.addWidget(self.sidebar_toggle_button)
        self.main_layout.addWidget(self.splitter)

        # --- Left Sidebar (Column 1) ---
        self.left_sidebar_widget = QWidget()
        self.left_sidebar_layout = QVBoxLayout(self.left_sidebar_widget)
        self.left_sidebar_layout.setContentsMargins(5, 5, 5, 5)
        self.left_sidebar_layout.setSpacing(10)
        self.left_sidebar_widget.setStyleSheet("background-color: #EAEAEA;") # Sidebar background

        # Placeholder for Search Bar (can be QLineEdit)
        self.search_bar = QPushButton("[ Search Bar Placeholder ]") # Using button as placeholder
        self.search_bar.setEnabled(False) # Disable placeholder

        # New Task Button
        self.new_task_button = QPushButton("+ New task") # Add icon later if needed
        self.new_task_button.setObjectName("newTaskButton") # For specific styling
        # self.new_task_button.setIcon(QIcon("path/to/plus_icon.png")) # Example icon
        self.new_task_button.setToolTip("Create a new agent task (Ctrl+K)") # Tooltip like image
        self.new_task_button.setShortcut("Ctrl+K") # Implement the shortcut

        # Task List
        self.task_list_widget = QListWidget()
        # Add dummy items for visual representation
        QListWidgetItem("AI Struggles with Codin...", self.task_list_widget)
        QListWidgetItem("Another Task Example", self.task_list_widget)
        self.task_list_widget.setCurrentRow(0) # Select the first item

        self.left_sidebar_layout.addWidget(self.search_bar)
        self.left_sidebar_layout.addWidget(self.new_task_button)
        self.left_sidebar_layout.addWidget(self.task_list_widget, 1) # List takes available space

        self.splitter.addWidget(self.left_sidebar_widget)

        # Connect New Task button
        self.new_task_button.clicked.connect(self.clear_task_state)

        # --- Middle Panel (Column 2: Agent Interaction) ---
        self.middle_panel_widget = QFrame() # Use QFrame for potential border
        self.middle_panel_widget.setObjectName("middleFrame")
        self.middle_panel_widget.setFrameShape(QFrame.NoFrame) # No frame initially, styled by CSS
        self.middle_panel_layout = QVBoxLayout(self.middle_panel_widget)
        self.middle_panel_layout.setContentsMargins(10, 10, 10, 10) # Padding inside middle panel
        self.middle_panel_layout.setSpacing(8)

        # Top Bar (Working Dir - moved here conceptually, button remains global for now)
        self.working_dir_label = QLabel("Working Directory: Not Selected")
        self.working_dir_button = QPushButton("Select Working Directory")
        self.working_dir_button.clicked.connect(self.select_working_directory)
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(self.working_dir_label)
        dir_layout.addStretch()
        dir_layout.addWidget(self.working_dir_button)

        # Agent Log / Interaction Area
        self.output_log_label = QLabel("Agent Interaction Log:") # Renamed
        self.output_log = QTextBrowser() # Initialize the output log widget HERE
        self.output_log.setObjectName("logBrowser") # For specific styling
        self.output_log.setReadOnly(True)
        # self.output_log.setStyleSheet("background-color: #FFFFFF; color: #333333;") # Light theme log

        # Task Input Area (Bottom)
        self.task_input_label = QLabel("Enter Task Description or Message:") # Renamed
        self.task_input = QTextEdit()
        self.task_input.setPlaceholderText("e.g., 'Write a python script...' or 'Explain the error.'")
        self.task_input.setFixedHeight(80) # Adjusted height

        # Run/Send Button Area
        self.run_button = QPushButton("Run Task / Send") # Renamed
        self.run_button.clicked.connect(self.start_agent_task)
        button_layout = QHBoxLayout()
        # Add attachment button placeholder if needed
        # self.attach_button = QPushButton("📎")
        # button_layout.addWidget(self.attach_button)
        button_layout.addStretch()
        button_layout.addWidget(self.run_button)

        # Add widgets to middle layout
        self.middle_panel_layout.addLayout(dir_layout)
        self.middle_panel_layout.addWidget(self.output_log_label)
        self.middle_panel_layout.addWidget(self.output_log, 1) # Log takes most space
        self.middle_panel_layout.addWidget(self.task_input_label)
        self.middle_panel_layout.addWidget(self.task_input)
        self.middle_panel_layout.addLayout(button_layout)

        self.splitter.addWidget(self.middle_panel_widget)

        # --- Right Panel (Column 3: File/Computer View) ---
        self.file_viewer_widget = FileViewerWidget() # Instantiate the new widget
        self.splitter.addWidget(self.file_viewer_widget) # Add it to the splitter

        # --- Configure Splitter Sizes ---
        self.splitter.setSizes([150, 500, 550]) # Initial sizes (adjust as needed)
        self.splitter.setStretchFactor(0, 0) # Left sidebar fixed width initially
        self.splitter.setStretchFactor(1, 1) # Middle panel stretches
        self.splitter.setStretchFactor(2, 1) # Right panel stretches

        # --- Status Bar & Progress ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("Status: Idle")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0) # Indeterminate progress
        self.status_bar.addPermanentWidget(self.status_label)
        self.status_bar.addPermanentWidget(self.progress_bar, 1) # Stretch factor 1

        self.setup_logging() # Setup logging framework - MOVED HERE

    def setup_logging(self):
        """Configures the Python logging framework to use the GUI handler."""
        self.log_handler = GUILogHandler()
        # Define a custom level for SUCCESS if needed, mapping it to an integer value
        logging.addLevelName(logging.INFO + 5, "SUCCESS")

        # Format logs
        log_format = "%(message)s" # Keep it simple for the GUI display
        formatter = logging.Formatter(log_format)
        self.log_handler.setFormatter(formatter)

        # Connect the handler's signal to the GUI update slot
        self.log_handler.log_signal.connect(self.log_message_to_gui)

        # Get the root logger and add the handler
        logger = logging.getLogger() # Get root logger
        # Remove existing handlers to avoid duplicates if this is called multiple times
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        logger.addHandler(self.log_handler)
        logger.setLevel(logging.DEBUG) # Set the desired logging level (e.g., DEBUG, INFO)
        logging.info("Logging initialized.") # Initial log message

    # --- Method Definitions ---

    @pyqtSlot() # Explicitly mark as a slot connected to button click
    def clear_task_state(self):
        """Clears the UI elements related to the current task and stops any running worker."""
        logging.info("Clearing current task state...")

        # Stop existing worker if running (QThreadPool doesn't offer direct 'stop',
        # rely on the worker's internal 'is_running' flag and signal handling)
        # We might need a way to track active workers if multiple can run,
        # but for now, assume only one task runs at a time.
        # The 'is_running' flag in AgentWorker helps prevent signals after stopping.
        # QThreadPool manages thread cleanup automatically.
        # We might need to explicitly stop the worker logic if it's long-running.
        # For now, we'll rely on the worker checking its 'is_running' flag.
        logging.warning("Clearing task state. Any running background task will be orphaned if not stopped internally.")
        # TODO: Implement a more robust way to signal cancellation to QRunnables in the pool if needed.

        # Clear UI elements
        self.output_log.clear()
        self.task_input.clear()
        self.file_viewer_widget.clear_display() # Clear the file viewer
        self.update_status("Idle")
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True) # Re-enable run button
        self.task_input.setEnabled(True) # Re-enable input

        # Optionally reset working directory or keep it
        if self.working_directory:
            logging.info(f"Working directory remains: {self.working_directory}")
        else:
            logging.info("No working directory selected.")

        logging.info("Ready for new task.")

    @pyqtSlot() # Explicitly mark as a slot connected to button click
    def toggle_sidebar(self):
        """Toggles the visibility of the left sidebar."""
        if self.left_sidebar_widget.isVisible():
            self.left_sidebar_widget.setVisible(False)
            self.sidebar_toggle_button.setText(">")
        else:
            self.left_sidebar_widget.setVisible(True)
            self.sidebar_toggle_button.setText("<")

    @pyqtSlot() # Explicitly mark as a slot connected to button click
    def select_working_directory(self):
            directory = QFileDialog.getExistingDirectory(self, "Select Working Directory")
            if directory:
                self.working_directory = Path(directory)
                self.working_dir_label.setText(f"Working Directory: {self.working_directory}")
                logging.info(f"Working directory set to: {self.working_directory}")
            else:
                logging.info("Working directory selection cancelled.")


    @pyqtSlot() # Explicitly mark as a slot connected to button click
    def start_agent_task(self):
            if not self.working_directory:
                logging.warning("Please select a working directory first.")
                # Optionally show a message box:
                # from PyQt5.QtWidgets import QMessageBox
                # QMessageBox.warning(self, "Missing Directory", "Please select a working directory first.")
                return

            task = self.task_input.toPlainText().strip()
            if not task:
                logging.warning("Please enter a task description.")
                # Optionally show a message box
                return
            # Basic Input Validation Example: Check length
            if len(task) > 2000: # Arbitrary limit
                 logging.warning("Task description is very long. Please keep it concise.")
                 # Optionally show a message box
                 return

            # Check if a task is already considered running (basic check)
            # QThreadPool manages actual threads, so we check our logical state.
            if not self.run_button.isEnabled(): # Use button state as proxy for running task
                logging.warning("A task is already running.")
                return

            logging.info(f"Starting new task: {task}")
            self.run_button.setEnabled(False)
            self.task_input.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.status_label.setText("Status: Running...")

            # Create worker (QRunnable) and pass the signals object
            worker = AgentWorker(task, self.working_directory, self.worker_signals)

            # Connect signals from the GUI's signal object
            # Disconnect previous connections first if reusing signals object heavily,
            # but for single task runs, it might be okay. Let's disconnect for safety.
            try:
                # self.worker_signals.log_message.disconnect() # Not used directly anymore
                self.worker_signals.task_complete.disconnect()
                self.worker_signals.status_update.disconnect()
                self.worker_signals.file_focus_update.disconnect()
                self.worker_signals.file_content_update.disconnect()
            except TypeError: # Ignore if not connected
                pass

            # Connect signals for this run
            # self.worker_signals.log_message.connect(self.log_message_to_gui) # Handled by logging framework now
            self.worker_signals.task_complete.connect(self.on_task_complete)
            self.worker_signals.status_update.connect(self.update_status)
            # Connect file signals to the FileViewerWidget slots
            self.worker_signals.file_focus_update.connect(self.file_viewer_widget.set_file_to_display)
            self.worker_signals.file_content_update.connect(self.file_viewer_widget.update_displayed_content)

            # Start the worker in the global thread pool
            QThreadPool.globalInstance().start(worker)

    @pyqtSlot(str, str) # Explicitly mark as a slot
    def log_message_to_gui(self, message: str, level_name: str):
            """Slot to append a formatted log message from the logging framework."""
            color = QColor("#333333") # Default text color
            prefix = "" # Prefix is now part of the formatted message from handler if desired
            level_upper = level_name.upper() # Use level_name passed by handler

            if level_upper == "ERROR":
                color = QColor("#D32F2F") # Red
                prefix = "❌ ERROR: " # Keep prefix for visual cue
            elif level_upper == "WARNING":
                color = QColor("#FFA000") # Amber
                prefix = "⚠️ WARNING: "
            elif level_upper == "SUCCESS": # Custom level name
                color = QColor("#388E3C") # Green
                prefix = "✅ SUCCESS: "
            elif level_upper == "DEBUG":
                color = QColor("#1976D2") # Blue
                prefix = "🐞 DEBUG: "
            elif level_upper == "INFO":
                color = QColor("#333333") # Default info color
                prefix = "ℹ️ INFO: "
            # Add other levels if needed

            self.output_log.setTextColor(color)
            self.output_log.append(f"{prefix}{message}")
            self.output_log.ensureCursorVisible() # Auto-scroll
            self.output_log.moveCursor(QTextCursor.End)

    @pyqtSlot(str) # Explicitly mark as a slot
    def update_status(self, status_text: str):
            """Updates the status label in the status bar."""
            self.status_label.setText(f"Status: {status_text}")

    # update_file_display and update_file_content_display are now methods of FileViewerWidget

    @pyqtSlot(str) # Explicitly mark as a slot
    def on_task_complete(self, final_result: str):
            """Handles UI updates when the task finishes."""
            logging.info("--- Task Finished ---")
            # Optionally display final result summary if needed
            if final_result and not final_result.startswith("Error:"):
                logging.log(logging.INFO + 5, f"Final Result: {final_result}") # Use SUCCESS level
            elif final_result: # Log errors if any result string starts with Error:
                logging.error(f"Task completed with error: {final_result}")

            self.run_button.setEnabled(True)
            self.task_input.setEnabled(True)
            # Ensure UI updates happen before clearing references
            self.progress_bar.setVisible(False)
            self.update_status("Idle")
            # No explicit cleanup needed for QThreadPool threads


    def closeEvent(self, event):
            """Ensure worker threads are handled cleanly on window close."""
            logging.info("Close event triggered. Cleaning up...")
            # QThreadPool management:
            # 1. Signal any running workers to stop (via their internal flags).
            #    This requires tracking active workers or iterating through the pool,
            #    which is complex. The AgentWorker's `stop()` method and `is_running`
            #    flag are crucial here. We assume workers check this flag periodically.
            # 2. Wait for threads to finish.

            # Disconnect and close log handler *before* waiting for threads,
            # as thread completion might trigger final log messages during shutdown.
            logger = logging.getLogger()
            try:
                logger.removeHandler(self.log_handler)
                self.log_handler.close() # Explicitly close the handler
                self.log_handler.log_signal.disconnect()
                logging.info("GUI Log Handler removed and closed.")
            except Exception as e: # Catch potential errors during handler cleanup
                 logging.error(f"Error removing/closing GUI log handler: {e}")

            # For simplicity, we'll just wait for the pool to finish active tasks.
            # This might block the GUI closing if tasks don't finish quickly.
            # A more robust solution might involve explicitly signaling cancellation.
            logging.warning("Waiting for active tasks in thread pool to complete...")
            QThreadPool.globalInstance().waitForDone() # Wait for all tasks
            logging.info("Thread pool tasks finished.")

            event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Apply a global style or fusion style for better cross-platform look
    # app.setStyle("Fusion")
    main_window = AgentAppGUI()
    main_window.show()
    exit_code = app.exec_()
    # Explicitly delete the handler instance before logging shutdown and exit
    try:
        del main_window.log_handler
        logging.info("GUI Log Handler instance deleted.")
    except AttributeError:
        logging.warning("Could not delete main_window.log_handler (already deleted or never created?).")
    except Exception as e:
        logging.error(f"Error deleting GUI log handler instance: {e}")

    logging.shutdown() # Explicitly shutdown logging before sys.exit
    sys.exit(exit_code)