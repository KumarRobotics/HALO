import os
import datetime
from pathlib import Path


class EvaluationLogger:
    """
    Simple logger class for ROS2 applications.
    """
    
    def __init__(self, log_path: str, mission: str):
        """
        Initialize the EvaluationLogger.
        
        Args:
            log_path (str): Directory path where log files will be saved
            mission (str): Mission name for the logging session
        """
        self.log_path = Path(log_path)
        self.mission = mission
        
        # Create directory if it doesn't exist
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_path / f"{mission}_{timestamp}.txt"
        
        # Write initial header
        with open(self.log_file, 'a') as f:
            f.write(f"=== {mission} - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
    
    def log(self, data, label=None):
        """
        Log data to the file.
        
        Args:
            data: Data to log (any type)
            label: Optional label for the data
        """
        with open(self.log_file, 'a') as f:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            
            if label:
                f.write(f"[{timestamp}] {label}: {data}\n")
            else:
                f.write(f"[{timestamp}] {data}\n")


# Example usage
if __name__ == "__main__":
    logger = EvaluationLogger("/tmp/logs", "test_mission")
    
    logger.log("Mission started")
    logger.log({"x": 1.5, "y": 2.3}, "Robot Position")
    logger.log([0.5, 1.2, 0.8], "Sensor Data")