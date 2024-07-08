# Navigation-with-drl

1. Install Docker image:
    ```sh
    docker pull chalkchalk/ur5_pybullet_moveit:pytorch
    ```

2. Enable GUI forwarding:
    ```sh
    xhost +
    ```

3. Start Docker:
    ```sh
    docker run -dit\
    --name ur5_pybullet_moveit \
    --cap-add=SYS_PTRACE \
    --shm-size=8G \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /home/$USER/docker_mount/ur5_pybullet_ros/src:/root/catkin_ws/src \
    -e DISPLAY=unix$DISPLAY \
    --privileged \
    --network=host\
    chalkchalk/ur5_pybullet_moveit:pytorch
    ```

4. Start the Docker container:
    ```sh
    docker start ur5_pybullet_moveit
    docker exec -it ur5_pybullet_moveit bash
    cd /root/catkin_ws/
    catkin build
    roslaunch ur5_pybullet_ros ur5_pybullet_moveit_teb.launch
    ```

5. Start the Docker container on another terminal:
    ```sh
    docker start ur5_pybullet_moveit
    docker exec -it ur5_pybullet_moveit bash
    cd /root/catkin_ws/
    rosrun ur5_pybullet_ros task_controller.py
    ```
