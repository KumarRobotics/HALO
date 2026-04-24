#!/usr/bin/env python3
import sys
import rclpy
rclpy.init(args=None)
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
from rclpy.node import Node
from air_sem_gridmap_interfaces.srv import SetPrompt, SetTask, TerminationResult

from PyQt5.QtWidgets import QHBoxLayout, QCheckBox, QGroupBox
from PyQt5.QtCore import Qt, QTimer
import time

class SetPromptClient(Node):
    def __init__(self):
        super().__init__('set_prompt_client')
        self.cli = self.create_client(SetPrompt, 'set_prompt')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting...')
        self.req = SetPrompt.Request()

    def send_request(self, task_prompt, task_id):
        self.req.task_prompt = task_prompt
        self.req.task_id = int(task_id)
        future = self.cli.call_async(self.req)
        start_time = time.time()
        while not future.done() and (time.time() - start_time) < 5.0:
            rclpy.spin_once(self, timeout_sec=0.1)
        return future.result()

class SetTerminationClient(Node):
    def __init__(self):
        super().__init__('set_termination_client')
        self.cli = self.create_client(SetTask, 'set_termination')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting...')
        self.req = SetTask.Request()

    def send_request(self, task_prompt, task_id):
        self.req.task = task_prompt
        future = self.cli.call_async(self.req)
        start_time = time.time()
        while not future.done() and (time.time() - start_time) < 5.0:
            rclpy.spin_once(self, timeout_sec=0.1)
        return future.result()

class PromptWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Set Prompt Service Client')
        self.layout = QVBoxLayout()

        self.current_task_label = QLabel('Current Task: None (ID: 0)')
        self.layout.addWidget(self.current_task_label)

        # Horizontal layout for prompt and set_termination
        self.input_row = QHBoxLayout()
        # self.input_row.setSpacing(50)  # Increase spacing between columns

        # Prompt and SetTask labels on the same row
        self.prompt_label = QLabel('Task Prompt: (task1,task2)')
        self.termination_label = QLabel('Termination:')
        self.label_row = QHBoxLayout()
        self.label_row.addWidget(self.prompt_label)
        self.label_row.addWidget(self.termination_label)
        self.layout.addLayout(self.label_row)

        # Prompt and SetTask text boxes in the same row
        self.input_row = QHBoxLayout()
        self.prompt_input = QLineEdit()
        self.termination_input = QLineEdit()
        self.termination_input.setEnabled(False)
        self.input_row.addWidget(self.prompt_input)
        self.input_row.addWidget(self.termination_input)
        self.layout.addLayout(self.input_row)

        # Checkbox row: left empty, right is checkbox
        self.use_prompt_checkbox = QCheckBox('Use task prompt as termination')
        self.use_prompt_checkbox.setChecked(True)
        self.use_prompt_checkbox.stateChanged.connect(self.on_use_prompt_checkbox)
        self.checkbox_row = QHBoxLayout()
        self.checkbox_row.addWidget(self.use_prompt_checkbox)
        self.layout.addLayout(self.checkbox_row)


        # Buttons row: left column for New/Update, right for Set Termination
        buttons_row = QHBoxLayout()
        # Left column buttons
        left_buttons_col = QVBoxLayout()
        self.new_task_button = QPushButton('New Task')
        self.new_task_button.clicked.connect(self.new_task)
        left_buttons_col.addWidget(self.new_task_button)
        self.update_task_button = QPushButton('Update Task')
        self.update_task_button.clicked.connect(self.update_task)
        left_buttons_col.addWidget(self.update_task_button)
        buttons_row.addLayout(left_buttons_col)

        # Right column button
        right_buttons_col = QVBoxLayout()
        self.set_termination_button = QPushButton('Set\nTermination')
        self.set_termination_button.setEnabled(False)
        self.set_termination_button.clicked.connect(self.set_termination)
        right_buttons_col.addWidget(self.set_termination_button)
        buttons_row.addLayout(right_buttons_col)

        self.layout.addLayout(buttons_row)
        self.status_row = QHBoxLayout()

        # Task status box
        self.task_status_box = QGroupBox("Task Status")
        self.task_status_box_layout = QVBoxLayout()
        self.task_status_label = QLabel('Not Set')
        self.task_status_label.setWordWrap(True)
        self.task_status_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.task_status_box_layout.addWidget(self.task_status_label)
        self.task_status_box.setLayout(self.task_status_box_layout)
        self.task_status_box.setMinimumWidth(250)

        # Termination status box
        self.termination_status_box = QGroupBox("Termination Status")
        self.termination_status_box_layout = QVBoxLayout()
        self.termination_status_label = QLabel('Not Set')
        self.termination_status_label.setWordWrap(True)
        self.termination_status_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.termination_status_box_layout.addWidget(self.termination_status_label)
        self.termination_status_box.setLayout(self.termination_status_box_layout)
        self.termination_status_box.setMinimumWidth(250)

        self.status_row.addWidget(self.task_status_box, 1)
        self.status_row.addWidget(self.termination_status_box, 1)
        self.layout.addLayout(self.status_row)

        self.setLayout(self.layout)
        self.setMinimumSize(600, 350)

        self.current_task_prompt = None
        self.current_task_id = 0

        # Create TerminationResult service server
        self.termination_result_node = rclpy.create_node('termination_result_server')
        self.termination_result_server = self.termination_result_node.create_service(
            TerminationResult,
            'termination_result',
            self.termination_result_callback
        )
        
        # Use QTimer to periodically spin the node in the Qt event loop
        self.spin_timer = QTimer(self)
        self.spin_timer.timeout.connect(self.spin_ros_node)
        self.spin_timer.start(100)  # Spin every 100ms

    def spin_ros_node(self):
        if rclpy.ok():
            rclpy.spin_once(self.termination_result_node, timeout_sec=0.0)
    
    def closeEvent(self, event):
        # Stop QTimer and shutdown ROS2 cleanly
        self.spin_timer.stop()
        try:
            rclpy.shutdown()
        except Exception:
            pass
        event.accept()
    
    def termination_result_callback(self, request, response):
        # Display termination result in termination status label
        msg = f"TerminationResult: success={request.complete}, reason={request.description}"
        complete = True if request.complete else None
        self.set_termination_status(msg, success=complete)
        response.success = True
        return response

    def set_termination(self):
        # Only send termination (SetTask) service if available
        set_termination_text = self.termination_input.text()
        node = rclpy.create_node('service_checker')
        service_available = node.get_service_names_and_types()
        found = any(srv[0] == '/set_termination' for srv in service_available)
        node.destroy_node()
        if found:
            termination_client = SetTerminationClient()
            termination_response = termination_client.send_request(set_termination_text, self.current_task_id)
            if termination_response.success:
                self.set_termination_status('Termination set successfully.', success=True)
                self.termination_input.clear()
            else:
                fail_msg = f'Set Termination failed: {termination_response.message}'
                self.set_termination_status(fail_msg, success=False)
        else:
            self.set_termination_status('SetTermination service not available.', success=False)
    
    def on_use_prompt_checkbox(self, state):
        if self.use_prompt_checkbox.isChecked():
            self.termination_input.setEnabled(False)
            self.set_termination_button.setEnabled(False)
        else:
            self.termination_input.setEnabled(True)
            self.set_termination_button.setEnabled(True)

    def update_current_task_display(self):
        if self.current_task_prompt:
            self.current_task_label.setText(f'Current Task: {self.current_task_prompt} (ID: {self.current_task_id})')
        else:
            self.current_task_label.setText(f'Current Task: None (ID: {self.current_task_id})')


    def send_prompt(self, task_prompt, task_id):
        if not task_prompt or not isinstance(task_id, int):
            QMessageBox.warning(self, 'Input Error', 'Please enter a valid prompt.')
            return False
        if not (0 <= task_id <= 255):
            QMessageBox.warning(self, 'Input Error', 'Task ID must be an integer between 0 and 255.')
            return False
        
        self.set_task_status('Sending...', success=None)
        self.set_termination_status('Sending...', success=None)

        prompt_client = SetPromptClient()
        prompt_response = prompt_client.send_request(task_prompt, task_id)

        # Handle SetTermination only if service exists and checkbox is checked
        if self.use_prompt_checkbox.isChecked():
            node = rclpy.create_node('service_checker')
            service_available = node.get_service_names_and_types()
            found = any(srv[0] == '/set_termination' for srv in service_available)
            node.destroy_node()
            if found:
                termination_client = SetTerminationClient()
                termination_response = termination_client.send_request(task_prompt, task_id)
            else:
                termination_response = type('obj', (object,), {'success': True, 'message': 'Skipped, no service.'})()
        else:
            termination_response = type('obj', (object,), {'success': True, 'message': 'Skipped, not using prompt.'})()
        
        if prompt_response is None:
            prompt_response = type('obj', (object,), {'success': False, 'message': 'No response from prompt service.'})()
        if termination_response is None:
            termination_response = type('obj', (object,), {'success': False, 'message': 'No response from termination service.'})()
        
        success = prompt_response.success and termination_response.success

        if success:
            self.current_task_prompt = task_prompt
            self.current_task_id = task_id
            self.update_current_task_display()
            self.set_task_status('Service call succeeded.', success=True)
            self.set_termination_status(termination_response.message if hasattr(termination_response, 'message') else 'Termination succeeded.', success=True)
            return True
        else:
            fail_msg = 'Service call failed.'
            if not prompt_response.success:
                fail_msg += f' Prompt: {prompt_response.message}'
            if not termination_response.success:
                fail_msg += f' Task: {termination_response.message}'
            self.set_task_status(fail_msg, success=False)
            self.set_termination_status(termination_response.message if hasattr(termination_response, 'message') else 'Termination failed.', success=False)
            return False

    def set_task_status(self, text, success=None):
        self.task_status_label.setText(f'{text}')
        if success is None:
            color = 'black'
        else:
            color = 'green' if success else 'red'
        self.task_status_label.setStyleSheet(f'color: {color}; font-weight: bold;')

    def set_termination_status(self, text, success=None):
        self.termination_status_label.setText(f'{text}')
        if success is None:
            color = 'black'
        else:
            color = 'green' if success else 'red'
        self.termination_status_label.setStyleSheet(f'color: {color}; font-weight: bold;')

    def new_task(self):
        task_prompt = self.prompt_input.text()
        task_id = self.current_task_id + 1
        if self.send_prompt(task_prompt, task_id):
            self.prompt_input.clear()

    def update_task(self):
        task_prompt = self.prompt_input.text()
        task_id = self.current_task_id
        if self.send_prompt(task_prompt, task_id):
            self.prompt_input.clear()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PromptWindow()
    window.show()
    sys.exit(app.exec_())
