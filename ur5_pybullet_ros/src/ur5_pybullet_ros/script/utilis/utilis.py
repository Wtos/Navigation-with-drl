import pybullet as p

def get_joint_id(robot_id, joint_name):
    numJoints = p.getNumJoints(robot_id)
    for i in range(numJoints):
        info = p.getJointInfo(robot_id, i)
        jointID = info[0]
        jointName = info[1].decode("utf-8")
        if jointName == joint_name:
            return jointID
    raise ValueError("cannot find id!")