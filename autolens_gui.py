"""This code defines the GUI for AutoLens software."""

from PyQt5 import QtCore, QtGui, QtWidgets

class SquareLabel(QtWidgets.QLabel):
    """A QLabel subclass that maintains a square shape.
    
    This label widget automatically resizes itself to maintain a 1:1 aspect ratio,
    making it always appear as a perfect square. The size is determined by the 
    smaller of the width or height dimensions.
    """
    def __init__(self, *args, **kwargs):
        """Initialize the SquareLabel widget.
        
        Args:
            *args: Variable length argument list passed to QLabel
            **kwargs: Arbitrary keyword arguments passed to QLabel
        """
        super(SquareLabel, self).__init__(*args, **kwargs)
        self.setMinimumSize(1, 1)  # Set minimum dimensions to prevent collapse
    
    def resizeEvent(self, event: QtGui.QResizeEvent):
        """Handle resize events to maintain square shape.
        
        This method is called automatically when the widget is resized. It ensures
        the widget remains square by setting both width and height to the smaller
        of the two dimensions.
        
        Args:
            event (QtGui.QResizeEvent): The resize event containing the old and new sizes
        """
        # Keep the widget square and keep the aspect ratio
        size = min(self.width(), self.height())
        self.setFixedSize(size, size)
        super(SquareLabel, self).resizeEvent(event)

class InputDialog(QtWidgets.QDialog):
    """ Dialog to input lens design target specifications.
    """
    valueEntered = QtCore.pyqtSignal(tuple)
    def __init__(self, parent=None):
        super(InputDialog, self).__init__(parent)

        self.setWindowTitle("Input Values")
        
        layout = QtWidgets.QFormLayout(self)
        
        self.input1 = QtWidgets.QLineEdit(self)
        self.input1.setText('50')
        
        self.input2 = QtWidgets.QLineEdit(self)
        self.input2.setText('15')
        
        self.input3 = QtWidgets.QLineEdit(self)
        self.input3.setText('4.0')
        
        self.input4 = QtWidgets.QLineEdit(self)
        self.input4.setText('5')
        
        layout.addRow("Focal length (mm):", self.input1)
        layout.addRow("HFoV (deg):", self.input2)
        layout.addRow("F-number:", self.input3)
        layout.addRow("Element number:", self.input4)
        
        self.buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, self)
        self.buttonBox.accepted.connect(self.on_accept)
        self.buttonBox.rejected.connect(self.reject)
        
        layout.addWidget(self.buttonBox)

    def on_accept(self):
        values = (self.input1.text(), self.input2.text(), self.input3.text(), self.input4.text())
        self.valueEntered.emit(values)
        self.accept()

class OptimConstraintDialog(QtWidgets.QDialog):
    """Dialog to input optimization constraints."""
    
    valueEntered = QtCore.pyqtSignal(tuple)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Optimization Constraints")
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the dialog's UI components."""
        layout = QtWidgets.QFormLayout(self)
        
        # Center thickness constraint
        self.center_thickness = QtWidgets.QLineEdit(self)
        self.center_thickness.setText('0')
        layout.addRow("Center thickness (min):", self.center_thickness)
        
        # Edge thickness constraint
        self.edge_thickness = QtWidgets.QLineEdit(self)
        self.edge_thickness.setText('0')
        layout.addRow("Edge thickness (min):", self.edge_thickness)
        
        # Add button box
        self.buttonBox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            self
        )
        self.buttonBox.accepted.connect(self.on_accept)
        self.buttonBox.rejected.connect(self.reject)
        
        layout.addWidget(self.buttonBox)

    def on_accept(self):
        """Handle dialog acceptance and emit entered values."""
        values = (
            float(self.center_thickness.text()),
            float(self.edge_thickness.text())
        )
        self.valueEntered.emit(values)
        self.accept()

class PruneDialog(QtWidgets.QDialog):
    """Dialog to input surface pruning percentage."""
    
    valueEntered = QtCore.pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Prune Surface")
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the dialog's UI components."""
        layout = QtWidgets.QFormLayout(self)
        
        # Percentage input
        self.percentage = QtWidgets.QLineEdit(self)
        self.percentage.setText('10')
        layout.addRow("Pruning percentage (%):", self.percentage)
        
        # Add button box
        self.buttonBox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            self
        )
        self.buttonBox.accepted.connect(self.on_accept)
        self.buttonBox.rejected.connect(self.reject)
        
        layout.addWidget(self.buttonBox)

    def on_accept(self):
        """Handle dialog acceptance and emit entered value."""
        try:
            value = float(self.percentage.text())
            if 0 <= value <= 100:
                self.valueEntered.emit(value)
                self.accept()
            else:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Invalid Input",
                    "Percentage must be between 0 and 100."
                )
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Input",
                "Please enter a valid number."
            )

class OptimizationDialog(QtWidgets.QDialog):
    """Dialog to show optimization progress and stop button."""
    
    stopOptimization = QtCore.pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Optimization Progress")
        self._setup_ui()
        
        # Make dialog stay on top
        self.setWindowFlags(
            QtCore.Qt.Window |
            QtCore.Qt.CustomizeWindowHint |
            QtCore.Qt.WindowTitleHint |
            QtCore.Qt.WindowStaysOnTopHint  # Add this flag
        )
        
    def _setup_ui(self):
        """Setup the dialog's UI components."""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Status label
        self.status_label = QtWidgets.QLabel("Lens is optimizing...")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Stop button
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.clicked.connect(self.on_stop)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #ff4444;
                color: white;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #ff0000;
            }
        """)
        layout.addWidget(self.stop_button)
        
    def on_stop(self):
        """Handle stop button click."""
        self.status_label.setText("Stopping optimization...")
        self.stop_button.setEnabled(False)
        self.stopOptimization.emit()

class RenderProgressDialog(QtWidgets.QDialog):
    """Dialog to show rendering progress."""
    
    cancelRender = QtCore.pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Rendering Progress")
        self._setup_ui()
        
        # Make dialog stay on top
        self.setWindowFlags(
            QtCore.Qt.Window |
            QtCore.Qt.CustomizeWindowHint |
            QtCore.Qt.WindowTitleHint |
            QtCore.Qt.WindowStaysOnTopHint
        )
        
    def _setup_ui(self):
        """Setup the dialog's UI components."""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Progress label
        self.status_label = QtWidgets.QLabel("Rendering image...")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Cancel button
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.on_cancel)
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #ff4444;
                color: white;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #ff0000;
            }
        """)
        layout.addWidget(self.cancel_button)
        
    def on_cancel(self):
        """Handle cancel button click."""
        self.status_label.setText("Cancelling render...")
        self.cancel_button.setEnabled(False)
        self.cancelRender.emit()

class AutoLens_Window(object):
    """The main window of AutoLens software."""
    def create_window(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(900, 600)
        
        # Add window close behavior
        MainWindow.setWindowFlags(
            QtCore.Qt.Window |
            QtCore.Qt.WindowCloseButtonHint |
            QtCore.Qt.WindowMinimizeButtonHint |
            QtCore.Qt.WindowMaximizeButtonHint
        )

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout.setRowStretch(2, 2)

        self.create_menu_bar(MainWindow)
        self.create_central_widget(MainWindow)
        self.create_horizontal_widget(MainWindow)
        self.create_bottom_widget(MainWindow)
        self.create_status_bar(MainWindow)
        
    def create_menu_bar(self, MainWindow):
        """Create the menu bar."""
        # Create menu bar
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 900, 30))
        self.menubar.setObjectName("menubar")

        # 1. Lens menu
        self.menuLens = QtWidgets.QMenu(self.menubar)
        self.menuLens.setObjectName("menuLens")

        self.actionCreate = QtWidgets.QAction(MainWindow)
        self.actionCreate.setObjectName("actionCreate")
        self.menuLens.addAction(self.actionCreate)

        self.actionLoad = QtWidgets.QAction(MainWindow)
        self.actionLoad.setObjectName("actionLoad")
        self.menuLens.addAction(self.actionLoad)

        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        self.menuLens.addAction(self.actionSave)
        
        # 2. Optimization menu
        self.menuOptimization = QtWidgets.QMenu(self.menubar)
        self.menuOptimization.setObjectName("menuOptimization")

        self.actionStartDiff = QtWidgets.QAction(MainWindow)
        self.actionStartDiff.setObjectName("actionStartDiff")
        self.menuOptimization.addAction(self.actionStartDiff)

        self.actionStartCurriculum = QtWidgets.QAction(MainWindow)
        self.actionStartCurriculum.setObjectName("actionStartCurriculum")
        self.menuOptimization.addAction(self.actionStartCurriculum)

        self.menuOptimization.addSeparator()

        self.actionOptimConstraints = QtWidgets.QAction(MainWindow)
        self.actionOptimConstraints.setObjectName("actionOptimConstraints")
        self.menuOptimization.addAction(self.actionOptimConstraints)

        # 3. Analysis menu
        self.menuAnalysis = QtWidgets.QMenu(self.menubar)
        self.menuAnalysis.setObjectName("menuAnalysis")


        self.actionAnalyze = QtWidgets.QAction(MainWindow)
        self.actionAnalyze.setObjectName("actionAnalyze")
        self.menuAnalysis.addAction(self.actionAnalyze)

        self.actionPrune = QtWidgets.QAction(MainWindow)
        self.actionPrune.setObjectName("actionPrune")
        self.menuAnalysis.addAction(self.actionPrune)

        # 4. Image simulation menu
        self.menuImageSimulation = QtWidgets.QMenu(self.menubar)
        self.menuImageSimulation.setObjectName("menuImageSimulation")


        self.actionRender = QtWidgets.QAction(MainWindow)
        self.actionRender.setObjectName("actionRender")
        self.menuImageSimulation.addAction(self.actionRender)

        
        # Add menus to menu bar
        MainWindow.setMenuBar(self.menubar)
        self.menubar.addAction(self.menuLens.menuAction())
        self.menubar.addAction(self.menuOptimization.menuAction())
        self.menubar.addAction(self.menuAnalysis.menuAction())
        self.menubar.addAction(self.menuImageSimulation.menuAction())

        
    def create_central_widget(self, MainWindow):
        """Create the central widget."""
        self.data_table = QtWidgets.QTableWidget(self.centralwidget)
        self.data_table.setFont(QtGui.QFont("Arial", 8))
        self.data_table.setObjectName("data_table")
        self.data_table.setColumnCount(0)
        self.data_table.setRowCount(0)

        
        # Make all columns equal width
        header = self.data_table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        header.setDefaultSectionSize(60)  # Set default width for all columns
        
        self.gridLayout.addWidget(self.data_table, 1, 0, 1, 8)

        # Enable tooltips for truncated content
        self.data_table.setToolTip("")
        self.data_table.setToolTipDuration(-1)  # Show tooltip indefinitely until mouse moves

        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")

    def create_horizontal_widget(self, MainWindow):
        """Create the horizontal widget."""
        self.fig1 = SquareLabel(self.centralwidget)
        self.fig1.setStyleSheet("border: 1px solid black;")
        self.fig1.setText("")

        self.fig1.setObjectName("fig1")
        self.horizontalLayout.addWidget(self.fig1)

        self.fig2 = SquareLabel(self.centralwidget)
        self.fig2.setStyleSheet("border: 1px solid black;")
        self.fig2.setText("")
        self.fig2.setObjectName("fig2")
        self.horizontalLayout.addWidget(self.fig2)

        self.fig3 = SquareLabel(self.centralwidget)
        self.fig3.setStyleSheet("border: 1px solid black;")
        self.fig3.setText("")
        self.fig3.setObjectName("fig3")
        self.horizontalLayout.addWidget(self.fig3)

        self.gridLayout.addLayout(self.horizontalLayout, 2, 0, 1, 8)

    def create_bottom_widget(self, MainWindow):
        """Create the bottom widget."""
        self.bottomLayout = QtWidgets.QHBoxLayout()
        self.bottomLayout.setObjectName("bottomLayout")
        self.bottomLayout.setSpacing(10)  # Add spacing between items

        def add_separator():
            """Add a vertical separator line."""
            line = QtWidgets.QFrame()
            line.setFrameShape(QtWidgets.QFrame.VLine)
            line.setFrameShadow(QtWidgets.QFrame.Sunken)
            self.bottomLayout.addWidget(line)

        # Focal Length
        self.label1_name = QtWidgets.QLabel(self.centralwidget)
        self.label1_name.setText("FocLen:")
        self.label1_name.setObjectName("label1_name")
        self.bottomLayout.addWidget(self.label1_name)

        self.data1 = QtWidgets.QLabel(self.centralwidget)
        self.data1.setObjectName("data1")
        self.data1.setMinimumWidth(50)  # Set minimum width for data
        self.bottomLayout.addWidget(self.data1)

        add_separator()

        # Field of View
        self.label2_name = QtWidgets.QLabel(self.centralwidget)
        self.label2_name.setText("FoV:")
        self.label2_name.setObjectName("label2_name")
        self.bottomLayout.addWidget(self.label2_name)

        self.data2 = QtWidgets.QLabel(self.centralwidget)
        self.data2.setObjectName("data2")
        self.data2.setMinimumWidth(50)
        self.bottomLayout.addWidget(self.data2)

        add_separator()

        # F-number
        self.label3_name = QtWidgets.QLabel(self.centralwidget)
        self.label3_name.setText("F-number:")
        self.label3_name.setObjectName("label3_name")
        self.bottomLayout.addWidget(self.label3_name)

        self.data3 = QtWidgets.QLabel(self.centralwidget)
        self.data3.setObjectName("data3")
        self.data3.setMinimumWidth(50)
        self.bottomLayout.addWidget(self.data3)

        add_separator()

        # RMS Spot Size
        self.label4_name = QtWidgets.QLabel(self.centralwidget)
        self.label4_name.setText("RMS Spot Size:")
        self.label4_name.setObjectName("label4_name")
        self.bottomLayout.addWidget(self.label4_name)

        self.data4 = QtWidgets.QLabel(self.centralwidget)
        self.data4.setObjectName("data4")
        self.data4.setMinimumWidth(50)
        self.bottomLayout.addWidget(self.data4)

        add_separator()

        # Device
        self.label5_name = QtWidgets.QLabel(self.centralwidget)
        self.label5_name.setText("Device:")
        self.label5_name.setObjectName("label5_name")
        self.bottomLayout.addWidget(self.label5_name)

        self.data5 = QtWidgets.QLabel(self.centralwidget)
        self.data5.setObjectName("data5")
        self.data5.setMinimumWidth(100)  # Wider for GPU names
        self.bottomLayout.addWidget(self.data5)

        # Add stretching at the end to keep everything left-aligned
        self.bottomLayout.addStretch()

        self.gridLayout.addLayout(self.bottomLayout, 3, 0, 1, 8)

    def create_status_bar(self, MainWindow):
        """Create the status bar."""
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")

        MainWindow.setStatusBar(self.statusbar)

        self.retranslate_ui(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslate_ui(self, MainWindow):
        """Translate the UI text.
        """
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "AutoLens"))
        
        # Lens menu
        self.menuLens.setTitle(_translate("MainWindow", "Lens"))        
        self.actionCreate.setText(_translate("MainWindow", "Create lens"))
        self.actionLoad.setText(_translate("MainWindow", "Load lens"))
        self.actionSave.setText(_translate("MainWindow", "Save lens"))

        # Optimization menu
        self.menuOptimization.setTitle(_translate("MainWindow", "Optimize"))
        self.actionStartDiff.setText(_translate("MainWindow", "Differentiable Optimization"))
        self.actionStartCurriculum.setText(_translate("MainWindow", "Curriculum Optimization"))
        self.actionOptimConstraints.setText(_translate("MainWindow", "Optimization Constraints"))

        # Analysis menu
        self.menuAnalysis.setTitle(_translate("MainWindow", "Analysis"))
        self.actionAnalyze.setText(_translate("MainWindow", "Analyze"))
        self.actionPrune.setText(_translate("MainWindow", "Prune Surface"))

        # Image simulation menu
        self.menuImageSimulation.setTitle(_translate("MainWindow", "Image Simulation"))
        self.actionRender.setText(_translate("MainWindow", "Render"))

    def show_no_lens_warning(self):
        """Show warning dialog when no lens is created."""
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setText("No Lens Created")
        msg.setInformativeText("Please create or load a lens first.")
        msg.setWindowTitle("Warning")
        msg.exec_()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = AutoLens_Window()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())