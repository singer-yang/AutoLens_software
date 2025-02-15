"""Start AutoLens software from this code!

Note:
- Following the Qt style, we use camel case for function names.
- For DeepLens functions, we use snake case for function names.
"""
import sys
import os
import torch
import math
import cv2 as cv
import numpy as np
import threading
import time
import json
import os
import ctypes
import logging
from datetime import datetime

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QFileDialog, QTableWidgetItem, QHeaderView, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import pyqtSlot

from autolens_gui import AutoLens_Window, InputDialog, OptimConstraintDialog, PruneDialog, OptimizationDialog, RenderProgressDialog
from deeplens import GeoLens, create_camera_lens, create_cellphone_lens


class AutoLens_software(QMainWindow, AutoLens_Window):
    """AutoLens software class."""
    def __init__(self):
        super(AutoLens_software, self).__init__()

        # Initialize the software
        self.create_window(self)
        self.connectSignals()

        # Initialize the lens and constraints
        self.lens = None
        self.is_optimizing = False
        self.center_thickness_min = 0.0
        self.edge_thickness_min = 0.0
        
        # Add close event handler
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.optim_dialog = None
        self.render_thread = None
        self.render_dialog = None

    def closeEvent(self, event):
        """Handle the window close event"""
        if self.is_optimizing:
            self.funcOptimStop()
        
        # Accept the close event and quit the application
        event.accept()
        QtWidgets.QApplication.quit()

    def connectSignals(self):
        """Connect menu bar and data table signals to functions.
        """
        # Lens menu
        self.actionCreate.triggered.connect(self.funcShowDesignTargetDialog)
        self.actionLoad.triggered.connect(self.funcLoadLens)
        self.actionSave.triggered.connect(self.funcSaveLens)

        # Optimization menu
        self.actionStartDiff.triggered.connect(self.funcOptimDiff)
        self.actionStartCurriculum.triggered.connect(self.funcOptimDiff)
        self.actionOptimConstraints.triggered.connect(self.showOptimConstraintsDialog)

        # Analysis menu
        self.actionAnalyze.triggered.connect(self.funcAnalyzeLens)
        self.actionPrune.triggered.connect(self.showPruneDialog)

        # Image simulation menu
        self.actionRender.triggered.connect(self.funcRender)

        # Data table
        self.data_table.itemChanged.connect(self.onDataTableItemChanged)
        self.track_data_change = False

    

    # ========================================
    # Lens functions
    # ========================================
    def funcShowDesignTargetDialog(self):
        """ When clicking the create lens button, show the dialog to input design target values.
        """
        design_target_dialog = InputDialog(self)
        design_target_dialog.valueEntered.connect(self.handleDesignTargetValues)
        design_target_dialog.exec_()

    def handleDesignTargetValues(self, values):
        """ Handle the input design target values.
        """
        if values is None or not values:
            print("No values entered or dialog was cancelled.")
            return

        self.design_target = values
        self.createLens()

    def createLens(self):
        """Create a random lens with given design target values.

        """
        foclen = float(self.design_target[0])
        hfov = float(self.design_target[1])
        fnum = float(self.design_target[2])
        lens_num = int(self.design_target[3])
        imgh = foclen * math.tan(math.radians(hfov)) * 2
        if imgh < 15:
            self.lens = create_cellphone_lens(hfov=hfov, fnum=fnum, imgh=imgh, lens_num=lens_num)
        else:
            self.lens = create_camera_lens(foclen=foclen, imgh=imgh, fnum=fnum, lens_num=lens_num)

        # Load the lens into the table
        self.track_data_change = False
        self.updateLensTable()
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.track_data_change = True

        self.funcAnalyzeLens()

    def funcLoadLens(self):
        """Load lens from a local file. Can load json or zmx file.
        """
        filename1, _ = QFileDialog.getOpenFileName(self, 'Open files', './', 'Lens Files (*.json *.zmx)')
        if filename1:
            if filename1.endswith(('.json', '.zmx')):
                print('Lens loaded:', filename1)
                self.lens = GeoLens(filename=filename1)
                self.lens.write_lens_json('./temp_lens.json')
                
                self.track_data_change = False
                self.updateLensTable()
                self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
                self.track_data_change = True

                self.funcAnalyzeLens()
            else:
                QMessageBox.warning(self, "Invalid File Format", "Please select a file ending with .json or .zmx")
    
    def funcSaveLens(self):
        """Save lens data to a local file. Can save as json or zmx file.
        """
        if not self.check_lens_exists():
            return
            
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        save_path, _ = QFileDialog.getSaveFileName(self, "Save File", "", 
                                                 "JSON files (*.json);;ZMX files (*.zmx)")
        
        if save_path:
            if save_path.endswith('.json'):
                self.lens.write_lens_json(save_path)
                print('Lens data saved to:', save_path)
            elif save_path.endswith('.zmx'):
                self.lens.write_lens_zmx(save_path)
                print('Lens data saved to:', save_path)
    
    # ========================================
    # Optimization functions
    # ========================================
    def funcOptimDiff(self):
        """Create a thread to optimize the lens."""
        if not self.check_lens_exists():
            return
            
        # Create and show the optimization dialog
        self.optim_dialog = OptimizationDialog(self)
        self.optim_dialog.stopOptimization.connect(self.funcOptimStop)
        self.optim_dialog.show()
        
        # Start optimization in a separate thread
        self.is_optimizing = True  # Set flag before starting thread
        self.optim_thread = threading.Thread(target=self.funcOptimDiffCore)
        self.optim_thread.start()

    def funcOptimDiffCore(self):
        """Optimize the lens."""
        print('Start differentiable lens optimization...')
        result_dir = datetime.now().strftime("%m%d-%H%M%S")
        
        try:
            self.lens.optimize(result_dir=result_dir)
            self.optim_dir = './results/' + result_dir

            # After optimization, update the lens data and display the layout
            self.updateLensTable()
            self.funcAnalyzeLens()
        except SystemExit:
            print('Optimization stopped by user')
            # Perform cleanup if needed when stopped
            self.lens.analysis(save_name='./results/lens')
            self.displayLensLayout(self.fig1, './results/lens.png')
        finally:
            # Close the optimization dialog when done
            if self.optim_dialog:
                self.optim_dialog.close()
                self.optim_dialog = None
            self.is_optimizing = False

    def optimCurriculum(self):
        """Create a thread to optimize the lens. Because we donot want to block the main thread.
        """
        if not self.check_lens_exists():
            return
            
        self.optim_thread_curriculum = threading.Thread(target=self.optimCurriculumCore)
        self.optim_thread_curriculum.start()

    def optimCurriculumCore(self):
        """ Optimize the lens.
        """
        print('Start curriculum lens optimizatoin...')
        self.is_optimizing = True
        result_dir = datetime.now().strftime("%m%d-%H%M%S")
        # TODO: implement curriculum optimization
        self.lens.optimize_curriculum(result_dir = result_dir)
        self.optim_dir = './results/' + result_dir

        # After optimization, update the lens data and display the layout
        self.updateLensTable()
        self.funcAnalyzeLens()

    def funcOptimStop(self):
        """When the stop optimization button is clicked, stop the optimization thread."""
        if not self.check_lens_exists():
            return
            
        if self.optim_thread is not None and self.optim_thread.is_alive():
            print('Stopping optimization...')
            # Stop the optimization thread
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(self.optim_thread.ident), 
                ctypes.py_object(SystemExit)
            )
            self.optim_thread = None
            self.is_optimizing = False
            
            # Close the optimization dialog
            if self.optim_dialog:
                self.optim_dialog.close()
                self.optim_dialog = None

    def showOptimConstraintsDialog(self):
        """Show dialog to set optimization constraints."""
        if not self.check_lens_exists():
            return
            
        dialog = OptimConstraintDialog(self)
        dialog.center_thickness.setText(str(self.center_thickness_min))
        dialog.edge_thickness.setText(str(self.edge_thickness_min))
        dialog.valueEntered.connect(self.handleOptimConstraints)
        dialog.exec_()

    def handleOptimConstraints(self, values):
        """Handle the optimization constraint values."""
        self.center_thickness_min = values[0]
        self.edge_thickness_min = values[1]
        
        # Update lens parameters with new constraints
        if self.lens is not None:
            raise NotImplementedError('Changing optimization constraints is not supported yet.')
            self.lens.center_thickness_min = self.center_thickness_min
            self.lens.edge_thickness_min = self.edge_thickness_min
            print(f"Updated optimization constraints: center_thickness_min={self.center_thickness_min}, edge_thickness_min={self.edge_thickness_min}")

    # ========================================
    # Analysis functions
    # ========================================
    def funcAnalyzeLens(self):
        """Analysis the lens and draw lens layout."""
        if not self.check_lens_exists():
            return
            
        # Lens analysis
        self.lens.write_lens_json('./results/lens.json')
        self.lens.analysis(save_name='./results/lens')

        # Display lens layout
        self.displayLensLayout(self.fig1, './results/lens.png')
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        self.updateLensInfo()

    def showPruneDialog(self):
        """Show dialog to set surface pruning percentage."""
        if not self.check_lens_exists():
            return
            
        dialog = PruneDialog(self)
        dialog.valueEntered.connect(self.handlePruneValue)
        dialog.exec_()

    def handlePruneValue(self, percentage):
        """Handle the pruning percentage value."""
        try:
            raise NotImplementedError('Pruning is not supported yet.')
            self.lens.prune_surface(percentage / 100.0)  # Convert percentage to decimal
            
            # Update the display after pruning
            self.updateLensTable()
            self.funcAnalyzeLens()
            
            print(f"Surface pruned with {percentage}% threshold")

        except Exception as e:
            QMessageBox.warning(
                self,
                "Pruning Error",
                f"Error during surface pruning: {str(e)}"
            )

    # ========================================
    # Image simulation functions
    # ========================================
    def funcRender(self):
        """Render the image captured by the lens."""
        if not self.check_lens_exists():
            return
        
        # Create and show the rendering progress dialog
        self.render_dialog = RenderProgressDialog(self)
        self.render_dialog.cancelRender.connect(self.cancelRender)
        self.render_dialog.show()
        
        # Start rendering in a separate thread
        self.render_thread = threading.Thread(target=self.renderCore)
        self.render_thread.start()

    def renderCore(self):
        """Core rendering function that runs in a separate thread."""
        try:
            # Render the image
            save_name = "./results/lens_render"
            img_org = cv.cvtColor(cv.imread(f'./datasets/resolution_chart1.png'), cv.COLOR_BGR2RGB)
            img_render = self.lens.render_single_img(img_org, spp=16, unwarp=False, save_name=save_name)

            # Display the rendered image directly from tensor
            self.displayImgRender(img_render)
            
        except Exception as e:
            if not isinstance(e, SystemExit):  # Don't show error if cancelled
                QMessageBox.warning(
                    self,
                    "Rendering Error",
                    f"Error during rendering: {str(e)}"
                )
        finally:
            # Close the progress dialog
            if self.render_dialog:
                self.render_dialog.close()
                self.render_dialog = None
            self.render_thread = None

    def cancelRender(self):
        """Cancel the rendering process."""
        if self.render_thread and self.render_thread.is_alive():
            print('Cancelling render...')
            # Stop the render thread
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(self.render_thread.ident), 
                ctypes.py_object(SystemExit)
            )
            self.render_thread = None
            
            # Close the render dialog
            if self.render_dialog:
                self.render_dialog.close()
                self.render_dialog = None

    def displayImgRender(self, img_render):
        """Display the rendered image of the lens in the third window (fig3).
        
        Args:
            img_render (numpy.ndarray): The rendered image of the lens. Shape: [H,W,3], range: [0,255]
        """
        try:

            # img_render is already a numpy array [H,W,3] 
            img_np = img_render.astype(np.uint8)  # Scale to 0-255 range
            
            height, width, channel = img_np.shape
            bytes_per_line = 3 * width
            
            # Convert numpy array to QImage
            q_img = QtGui.QImage(img_np.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            
            # Scale the image to fit the window while maintaining aspect ratio
            size = self.fig3.size()
            scaled_pixmap = pixmap.scaled(size, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            
            # Display the image
            self.fig3.setPixmap(scaled_pixmap)
            self.fig3.setScaledContents(True)
        except Exception as e:
            # If image display fails, clear the display and show error
            self.fig3.clear()
            self.fig3.setText("Error displaying render")
            print(f"Error displaying render: {str(e)}")

    # ========================================
    # Data table functions
    # ========================================
    def onDataTableItemChanged(self, item):

        """ Update lens data when table item changed.
        """
        if not self.check_lens_exists():
            return
            
        if self.track_data_change:

            row = item.row()
            column = item.column()

            if column == 1:
                self.lens.surfaces[row].c = 1 / torch.Tensor([float(item.text())]).to(self.lens.device)
            elif column == 2:
                self.lens.surfaces[row].r = float(item.text())
            elif column == 3:
                for i in range(row + 1, len(self.lens.surfaces)):
                    self.lens.surfaces[i].d += float(item.text())
                self.lens.d_sensor += float(item.text())
            elif column == 4:
                raise NotImplementedError('Changing glass material is not supported yet.')
            elif column == 5:
                assert self.lens.surfaces[row].__class__.__name__ == 'Aspheric', 'Only aspheric surface has conic constant.'
                self.lens.surfaces[row].k = torch.Tensor([float(item.text())]).to(self.lens.device)
            elif column >= 6:
                assert self.lens.surfaces[row].__class__.__name__ == 'Aspheric', 'Only aspheric surface has aspheric coefficients.'
                ai_idx = (column - 6) * 2
                self.lens.surfaces[row].__setattr__(f'ai{ai_idx}', torch.Tensor([float(item.text())]).to(self.lens.device))

            self.lens.analysis(save_name='./results/lens')
            self.displayLensLayout(self.fig1, './results/lens.png')


    # ========================================
    # Backend functions
    # ========================================
    def check_lens_exists(self):
        """Check if lens exists, show warning if not."""
        if self.lens is None:
            self.show_no_lens_warning()
            return False
        return True
    
    def updateLens(self):
        """Update all the lens data and display. Currently not used.
        """
        # Update the numerical data
        self.updateLensTable()
        self.updateLensInfo()

        # Update the display
        self.displayLensLayout(self.fig1, './results/lens.png')
        self.displayPSF()
        # self.displayImgRender()
        # self.displayImgRec()

    def updateLensTable(self):
        """Update the lens data in the table.
        """
        # Clean the table
        self.data_table.clearContents()

        # Create a new table
        lens = self.lens
        self.data_table.setColumnCount(14)
        self.data_table.setHorizontalHeaderLabels(['Type', 'RoC', 'Semi-diameter', 'Thickness', 'Glass-next', 'Conic', 'ai2', 'ai4', 'ai6', 'ai8', 'ai10', 'ai12', 'ai14', 'ai16'])
        self.data_table.setRowCount(len(lens.surfaces))
        self.data_table.setVerticalHeaderLabels(['Surf ' + str(i + 1) for i in range(len(lens.surfaces))])
        
        for surf_idx, surf in enumerate(lens.surfaces):
            # Shared parameters
            surf_type = surf.__class__.__name__
            self.data_table.setItem(surf_idx, 0, QTableWidgetItem(str(surf_type)))
            self.data_table.setItem(surf_idx, 2, QTableWidgetItem(str(format(surf.r, '.2f'))))
            d_next = lens.surfaces[surf_idx + 1].d - surf.d if surf_idx < len(lens.surfaces) - 1 else lens.d_sensor - surf.d
            self.data_table.setItem(surf_idx, 3, QTableWidgetItem(str(format(d_next.item(), '.2f'))))
            if surf.mat2.name != 'air':
                self.data_table.setItem(surf_idx, 4, QTableWidgetItem(str(surf.mat2.name)))
            
            # Specific parameters
            if surf_type == 'Spheric':
                roc = format(1 / surf.c.item(), '.2f') if surf.c.item() != 0 else np.inf
                self.data_table.setItem(surf_idx, 1, QTableWidgetItem(str(roc)))
                
            elif surf_type == 'Aperture':
                pass
                
            elif surf_type == 'Aspheric':
                roc = format(1 / surf.c.item(), '.2f') if surf.c.item() != 0 else np.inf
                self.data_table.setItem(surf_idx, 1, QTableWidgetItem(str(roc)))
                self.data_table.setItem(surf_idx, 5, QTableWidgetItem(str(format(surf.k.item(), '.2f'))))
                for i in range(1, surf.ai_degree + 1):
                    ai_value = format(eval(f'surf.ai{i*2}.item()'), '.2e')
                    self.data_table.setItem(surf_idx, 5 + i, QTableWidgetItem(str(ai_value)))
                                  
            else:
                raise NotImplementedError
            
        self.data_table.resizeColumnsToContents()

    def updateLensInfo(self):
        """Display the info of the lens at the bottom of the window."""
        lens = self.lens
        self.data1.setText(str(format(lens.foclen, '.1f')))
        self.data2.setText(str(format(math.degrees(lens.hfov), '.1f')))
        self.data3.setText(str(format(lens.fnum, '.1f')))
        # self.data4.setText(str(format(lens.rms_spot_size, '.2f')))
        self.data4.setText(str(0.0))
        
        # Show device info
        device_name = str(lens.device).upper()
        if device_name == 'CPU':
            self.data5.setText(device_name)
        else:
            gpu_props = torch.cuda.get_device_properties(lens.device)
            self.data5.setText(gpu_props.name)

    def displayLensLayout(self, la, dz):
        """Display the lens layout image in the first window (fig1).
        """
        pixmap = QPixmap(dz)
        size = la.size()
        scaled_pixmap = pixmap.scaled(size, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        la.setPixmap(scaled_pixmap)
        yu_di = QPixmap(dz)

        la.setPixmap(yu_di)
        la.setScaledContents(True)

    def displayPSF(self):
        """Display the PSF of the lens in the second window (fig2).
        """
        raise NotImplementedError

    def displayImgRec(self):
        """Display the reconstructed image after the network.
        """
        raise NotImplementedError

    
    
    def get_files_in_directory(self, directory):
        """Get files in the directory.
        """
        return set(os.listdir(directory))

    def monitor_directory(self, directory, interval=1):
        """Monitor the directory and load new lens layout and data."""
        previous_files = self.get_files_in_directory(directory)
        while True:
            time.sleep(interval)
            current_files = self.get_files_in_directory(directory)
            new_files = current_files - previous_files
            if new_files:
                for file in new_files:
                    print(f"新生成文件: {file}")
                    if file[-3:] == 'png':
                        time.sleep(2)
                        print("123", directory + '/' + file)
                        self.displayLensLayout(self.fig1, directory + '/' + file)
                    elif file[-4:] == 'json':
                        time.sleep(2)
                        print("789", directory + '/' + file)
                        self.updateLensTable()
                        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            previous_files = current_files
    
    
    def ztzs_pass(self):
        """Monitor the directory and load new lens layout and data."""
        result_dir = self.result_dir
        while True:
            try:
                self.displayLensLayout(self.fig1, result_dir + '/iter0.png')
                self.updateLensTable()
                self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
                break
            except:
                continue
        directory_to_monitor = result_dir
        self.monitor_directory(directory_to_monitor)

    



if __name__ == '__main__':
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    ui = AutoLens_software()
    ui.show()
    sys.exit(app.exec_())
