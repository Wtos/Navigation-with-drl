#!/usr/bin/env python3
import rospy
import numpy as np
from ros_wrapper_pkg.ros_msg.ros_dtype import ROSDtype
import time
import tf2_ros
import geometry_msgs.msg

def data_to_ros_msg(data, dtype:ROSDtype, ros_time, frame_id=""):
    return dtype.value.transform_rosmsg(data, ros_time, frame_id)

class RosWrapper:
    def __init__(self, rosnode_name): #if rosnode_name is empty, ros_init will not be called, since only one node for each script
        self.publishers = {} # name: [dtype, publisher]
        self.subscribers = {} # name: dtype, data
        self.rosnode_name = rosnode_name
        self.use_sim_time = True
        self.ros_time = 0
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        if rosnode_name:
            rospy.init_node(rosnode_name)
            rospy.loginfo("ROS wrapper init, node name = " + rosnode_name + ".")

    def add_publisher(self, topic, dtype, use_namespace=True, queue=5):
        full_topic = topic
        if use_namespace:
            full_topic = self.rosnode_name + '/' + topic
        pub = rospy.Publisher(full_topic, dtype.value.rosdtype, queue_size=queue)
        self.publishers[topic] = [dtype, pub]

    def add_subscriber(self, topic, dtype, data_handle=None, use_namespace=True, callback=None):
        """
        subsribe to a specific topic to update value for data_handle.
        @param data_handle: NOTE: must be mutable objects, including list, dict, and set, bytearray
        """
        # assert type(data_handle) in (list, dict, set, bytearray), "data_handle must be mutable objects, including list, dict, and set, bytearray."
        full_topic = topic
        if use_namespace:
            full_topic = self.rosnode_name + '/' + topic
        full_topic = "/" + full_topic
        if callback == None:
            callback = self.topic_callback
        rospy.Subscriber(full_topic, dtype.value.rosdtype, callback)
        self.subscribers[full_topic] = [dtype, data_handle]
        
    def topic_callback(self, msg):
        topic = msg._connection_header['topic']
        if self.subscribers[topic][0] == ROSDtype.FLOAT:
            self.subscribers[topic][1][0] = type(self.subscribers[topic][1][0])(msg.data)
        elif self.subscribers[topic][0] == ROSDtype.FLOAT_ARRAY:
            print(len(self.subscribers[topic][1][0]), len(list(msg.data)))
            assert len(self.subscribers[topic][1][0]) == len(list(msg.data)), "inbound float array leng"
            self.subscribers[topic][1][0] = list(msg.data)

    def publish_msg(self, topic, msg, frame_id=""):
        assert topic in self.publishers, "topic not registered!"
        self.publishers[topic][1].publish(data_to_ros_msg(msg, self.publishers[topic][0], self.ros_time, frame_id))
    
    def publish_tf(self, fram_id, child_frame, translation, rotation):
        transform_stamped = geometry_msgs.msg.TransformStamped()
        transform_stamped.header.frame_id = fram_id
        transform_stamped.child_frame_id = child_frame
        transform_stamped.header.stamp = rospy.Time.from_sec(self.ros_time)
        transform_stamped.transform.translation.x = translation[0]
        transform_stamped.transform.translation.y = translation[1]
        transform_stamped.transform.translation.z = translation[2]
        transform_stamped.transform.rotation.x = rotation[0]
        transform_stamped.transform.rotation.y = rotation[1]
        transform_stamped.transform.rotation.z = rotation[2]
        transform_stamped.transform.rotation.w = rotation[3]
        self.tf_broadcaster.sendTransform(transform_stamped)
    

if __name__ == "__main__":
    wrapper = RosWrapper("test_node")
    test_data = [[1,2]]
    wrapper.add_subscriber("test", ROSDtype.FLOAT_ARRAY, test_data)
    while not rospy.is_shutdown():
        time.sleep(0.1)
        print(test_data[0])
        
    