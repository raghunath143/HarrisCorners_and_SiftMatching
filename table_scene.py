"""
Implement the look_at_box_front() function in this python script
to capture an image of the box from its front side
"""

import os
import time
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

class TableYCBEnv():

    def __init__(self):

        self._renders = True
        self._egl_render = False
        self.connected = False

        self._window_width = 640
        self._window_height = 480
        self.object_uid = None
        self._timeStep = 1. / 1000.
        self.root_dir = os.path.dirname(os.path.abspath(__file__))

        self.connect()
        self.reset()


    def connect(self):
        """
        Connect pybullet.
        """
        if self._renders:
            self.cid = p.connect(p.SHARED_MEMORY)
            if (self.cid < 0):
                self.cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(0.6, 180.0, -41.0, [0.35, 0.58, 0.68])
        else:
            self.cid = p.connect(p.DIRECT)

        if self._egl_render:
            import pkgutil
            egl = pkgutil.get_loader("eglRenderer")
            if egl:
                p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")

        self.connected = True


    def reset(self):

        # Set the camera  .
        look = [0.1, 0.2, 0]
        distance = 2.5
        pitch = -56
        yaw = 245
        roll = 0.
        fov = 20.
        aspect = float(self._window_width) / self._window_height
        self.near = 0.1
        self.far = 10
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(look, distance, yaw, pitch, roll, 2)
        self._proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, self.near, self.far)
        self._light_position = np.array([-1.0, 0, 2.5])

        p.resetSimulation()
        p.setTimeStep(self._timeStep)
        p.setPhysicsEngineParameter(enableConeFriction=0)

        p.setGravity(0, 0, -9.81)
        p.stepSimulation()

        # Set table and plane
        plane_file = os.path.join(self.root_dir, 'data/floor/model_normalized.urdf') # _white
        table_file = os.path.join(self.root_dir, 'data/table/models/model_normalized.urdf')

        self.obj_path = [plane_file, table_file]
        self.plane_id = p.loadURDF(plane_file, [0, 0, 0])
        self.table_pos = np.array([0, 0, 0])
        self.table_id = p.loadURDF(table_file, self.table_pos[0], self.table_pos[1], self.table_pos[2],
                             0.707, 0., 0., 0.707)


    def _add_mesh(self, obj_file, trans, quat, scale=1):
        """
        Add a mesh with URDF file.
        """
        bid = p.loadURDF(obj_file, trans, quat, globalScaling=scale, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        return bid


    def place_objects(self, name):
        """
        Place of an object onto the table
        """
        
        # 3D translation is [tx, ty, tz]
        tx = 0
        ty = 0
        tz = 0.4

        # euler angles: roll, pitch, yaw
        # then convert roll, pitch, yaw to quaternion using function p.getQuaternionFromEuler()        
        roll = 0
        pitch = 0
        yaw = 0
        quaternion = p.getQuaternionFromEuler([roll, pitch, yaw])
        
        # put the box using the 3D translation and the 3D rotation
        urdf = os.path.join(self.root_dir, 'data', name, 'model_normalized.urdf')
        uid = self._add_mesh(urdf, [tx, ty, tz], [quaternion[0], quaternion[1], quaternion[2], quaternion[3]])  # xyzw
        self.object_uid = uid
        p.resetBaseVelocity(uid, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

        time.sleep(3.0)
        for _ in range(2000):
            p.stepSimulation()



    def get_observation(self, view_matrix):
        """
        Get observation and visualize
        """

        _, _, rgba, depth, mask = p.getCameraImage(width=self._window_width,
                                                   height=self._window_height,
                                                   viewMatrix=view_matrix,
                                                   projectionMatrix=self._proj_matrix,
                                                   physicsClientId=self.cid,
                                                   renderer=p.ER_BULLET_HARDWARE_OPENGL)
                                                   
        # visualization
        fig = plt.figure()
        
        # show RGB image
        ax = fig.add_subplot(1, 3, 1)
        plt.imshow(rgba[:, :, :3])
        ax.set_title('RGB image')
        
        # show depth image
        ax = fig.add_subplot(1, 3, 2)
        plt.imshow(depth)
        ax.set_title('depth image')
        
        # show segmentation mask
        ax = fig.add_subplot(1, 3, 3)
        plt.imshow(mask)
        ax.set_title('segmentation mask')                  
        plt.show()
                                                
                                                   
    # compute the view matrix to capture an image for the front of the cracker box
    # You need to first query the pose of the cracker box and then set the camera pose accordingly
    # Useful functions from pybullet: getBasePositionAndOrientation, getEulerFromQuaternion, computeViewMatrixFromYawPitchRoll
    # https://usermanual.wiki/Document/pybullet20quickstart20guide.479068914/html
    # Set the distance of the camera to 2.5
    def look_at_box_front(self):
        base_pos, orn = p.getBasePositionAndOrientation(self.object_uid)
        obj_euler = p.getEulerFromQuaternion(orn)
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=2.5,
            yaw=obj_euler[2]+90,
            pitch=obj_euler[1],
            roll=obj_euler[0],
            upAxisIndex=2)
        return view_matrix
        

# main function
if __name__ == '__main__':

    # create the table environment
    env = TableYCBEnv()
    
    # place the cracker box to the table
    name = '003_cracker_box'
    env.place_objects(name)

    # render image before looking at the box
    env.get_observation(env._view_matrix)
        
    # look at the box
    view_matrix = env.look_at_box_front()
    
    # render image again
    env.get_observation(view_matrix)
