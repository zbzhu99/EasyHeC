import numpy as np
import pink
import pinocchio as pin
from loop_rate_limiters import RateLimiter


if __name__ == "__main__":
    # Visualization
    robot = pin.RobotWrapper.BuildFromURDF(
        filename="./assets/xarm6_with_gripper.urdf",
        package_dirs=["./assets"],
        root_joint=None,
    )
    viz = pin.visualize.MeshcatVisualizer(
        robot.model, robot.collision_model, robot.visual_model
    )
    robot.setVisualizer(viz, init=False)
    viz.initViewer(open=False, loadModel=True)

    configuration = pink.Configuration(robot.model, robot.data, robot.q0)
    q = np.array([0, -0.349, -0.559, 0, 0.873, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    viz.display(q)

    rate = RateLimiter(frequency=10.0)
    dt = rate.period
    t = 0.0
    while True:
        viz.display(q)
        rate.sleep()
        t += dt
