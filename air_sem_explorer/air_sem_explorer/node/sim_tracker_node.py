import rclpy
import threading

from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from air_sem_explorer.planner.sim_tracker import SimTracker

def main(args=None) -> None:
    rclpy.init(args=args)

    executor = MultiThreadedExecutor()
    
    node = SimTracker()

    executor.add_node(node)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    try:
        executor_thread.start()
    finally:
        executor_thread.join()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
