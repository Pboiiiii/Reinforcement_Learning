import random
import pybullet as p
import pybullet_data
import numpy as np
import time

class Config:
    def __init__(self, model_path, render=False):
        self.model_path = model_path
        self.render = render

        self.modelID = None
        self.planeId = None
        self.forces = None
        self.numjoints = None
        self.prev_action = None
        self.info = {"Revolute":[], "Prismatic":[], "Spherical":[], "Planar":[], "Fixed":[]}

        # Reward function variables
        self.speed = None           # speed at the y-axis
        self.distance = None        # distance walked
        self.height = None          # keep robot to walk at a proper height
        self.x_error = None         # don't stray from the path at the y-axis
        self.step_count = 0         # time step for episode
        self.effort = None          # actuator effort

        self.walk_lenght = None


    def client_setup(self, debugger=False, iters=10):
        # Setting up the physics client
        cid = p.connect(p.SHARED_MEMORY)
        if cid < 0:
            p.connect(p.GUI if self.render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setPhysicsEngineParameter(numSolverIterations=iters)
        p.setTimeStep(1. / 120.)
        p.setGravity(0, 0, -9.81)

        if not debugger:
            # disable rendering during creation.
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)


    def model_setup(self, damping=0.05):
        self.modelID = p.loadURDF(self.model_path, [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]))
        self.planeId = p.loadURDF("plane.urdf", [0, 0, 0], [0, 0, 0, 1], useMaximalCoordinates=True)

        self.numjoints = p.getNumJoints(self.modelID)

        # Sorting every joint
        for joint in range(self.numjoints):
            if p.getJointInfo(self.modelID, joint)[2] == 0:self.info["Revolute"].append(joint)
            elif p.getJointInfo(self.modelID, joint)[2] == 1:self.info["Prismatic"].append(joint)
            elif p.getJointInfo(self.modelID, joint)[2] == 2:self.info["Spherical"].append(joint)
            elif p.getJointInfo(self.modelID, joint)[2] == 3:self.info["Planar"].append(joint)
            else:self.info["Fixed"].append(joint)

        # Adjusting damping of every joint to the desired value
        for joint in self.info["Revolute"]:
            p.changeDynamics(self.modelID, joint, linearDamping=damping, angularDamping=damping)


    def perform_action(self, action):
        """
        Performs the action (Applies torques to every joint):
            - Left ankle
            - Right ankle
            - Left knee
            - Right knee
            - Left hip
            - Right hip
        """
        mode = p.TORQUE_CONTROL

        action = np.array(action)
        self.forces = np.array([90, 80, 15, 90, 70, 15])

        # Clipping forces into a safe range
        self.forces = np.clip(action * self.forces, -100, 100)

        p.setJointMotorControlArray(
            bodyUniqueId=self.modelID,
            jointIndices=self.info["Revolute"],
            controlMode=mode,
            forces=self.forces
        )

        self.prev_action = self.forces


    def observation_info(self):
        """
        Updates the reward functions variables and
        returns the observation:
            - Body position               (y and z axis)
            - Body orientation            (all axis)
            - Body velocity               (all axis)
            - Body angular velocity       (all axis)
            - Joints angles               (all joints except for fixed ones)
            - Joints angular velocities   (all joints except for fixed ones)
            - Reaction forces between the ground and hips
            - Previous action
        """
        body_pos, body_orn = p.getBasePositionAndOrientation(self.modelID)
        body_vel, body_ang_rate = p.getBaseVelocity(self.modelID)

        obervation = [
            body_pos[1], body_pos[2],
            body_orn[0], body_orn[1], body_orn[2],
            body_vel[0], body_vel[1], body_vel[2],
            body_ang_rate[0], body_ang_rate[1], body_ang_rate[2]
        ]

        joints_state = p.getJointStates(self.modelID, self.info["Revolute"])
        joints_state = [state[:2] for state in joints_state] # Joints angles and angular velocities

        for joint in joints_state:
            obervation.append(joint[0])
            obervation.append(joint[1])

        contact_info1 = p.getContactPoints(self.modelID, self.planeId, 8, -1)
        contact_info2 = p.getContactPoints(self.modelID, self.planeId, 4, -1)
        nfs1, nfs2 = [cp[9] for cp in contact_info1], [cp[9] for cp in contact_info2]
        nf1, nf2 = max(nfs1) if len(nfs1) > 0 else 0.0, max(nfs2) if len(nfs2) > 0 else 0.0

        obervation.append(nf1)
        obervation.append(nf2)

        for force in self.prev_action:
            obervation.append(force)

        self.speed = body_vel[1]
        self.distance = body_pos[1]
        self.height = body_pos[2]
        self.x_error = body_pos[0]
        self.step_count += 1

        self.walk_lenght = body_pos[1]

        return obervation


    def reset(self):
        """
        Sets the robot to it's initial position,
        resets reward function variables and
        returns an initial observation:
        """
        p.resetBasePositionAndOrientation(self.modelID, [0, 0, 0.8], p.getQuaternionFromEuler([0, 0, 0]))
        for joint in self.info["Revolute"]:
            p.resetJointState(self.modelID, joint, 0, 0)

        self.speed = 0
        self.step_count = 0
        self.distance = 0
        self.height = 0.8
        self.x_error = 0

        observation = [
            0.0, 0.8,                                  # Body position
            0.0, 0.0, 0.0,                             # Body orientation
            0.0, 0.0, 0.0,                             # Body velocity
            0.0, 0.0, 0.0,                             # Body angular velocity
        ]

        for joint in range(len(self.info["Revolute"])):
            observation.append(0.0)                    # Joints angles
            observation.append(0.0)                    # Joints angular velocities

        observation.append(0.0)                        # Contact force 1
        observation.append(0.0)                        # Contact force 2

        for joint in range(len(self.info["Revolute"])):
            observation.append(0.0)                    # Previous action

        return observation


    def reward(self):
        """
        Returns the reward value of the current state, consisting of:
            - Current speed
            - Height of the main body
            - Distance between the robot and the path
            - Actual effort applied to the joints
        """
        #current_eff = sum(np.abs(self.effort) ** 2)

        R = (self.speed + self.distance * 25 - 1/self.height * 5 -
             (self.x_error**2) * 10)
        return R


    def step(self, action):
        """
        Performs the chosen action,
        retrieves the new state and reward and
        tells whether the state is terminal by:
            - Checking if many steps have passed and haven't surpassed a minimum threshold
            - Checking if the robot has fallen
        """
        # Performing action
        self.perform_action(action)

        # Getting the new state and reward value
        new_state = self.observation_info()
        reward = self.reward()

        # Checking if episode terminated
        terminated = False
        if self.height < 0.4 or self.height > 0.9:
            terminated = True
        elif self.step_count > 10000 and self.walk_lenght < 10:
            terminated = True

        # Additional information (not necessary)
        additional_info = {}

        return new_state, reward, terminated, additional_info


    def run_test(self):
        while True:
            p.stepSimulation()
            time.sleep(1. / 240.)
            action = [random.uniform(-1, 1) for i in range(len(self.info["Revolute"]))]
            #print(self.info["Revolute"])
            self.perform_action(action)

        #p.disconnect()


if __name__ == "__main__":
    config = Config("URDF_MODELS/test.urdf", render=True)
    config.client_setup(iters=50)
    config.model_setup()
    config.run_test()