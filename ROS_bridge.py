from pathlib import Path
import threading
import math

import rclpy
from std_msgs.msg import Float32MultiArray
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster

import Koncna_faza as base_sim


ROS_MODE = "ros_cmd_vel"

_original_init = base_sim.TankSimulator.__init__
_original_sample_energy = base_sim.TankSimulator._sample_energy_locked
_original_run = base_sim.TankSimulator.run
_original_lr_to_cmd_vel = base_sim.TankSimulator._left_right_command_to_cmd_vel


def _ros_cmd_vel_callback(self, msg: Twist):
    v_cmd = float(msg.linear.x)
    omega_cmd = float(msg.angular.z)

    self._ros_v_cmd = v_cmd
    self._ros_omega_cmd = omega_cmd

    active = abs(v_cmd) > 1e-4 or abs(omega_cmd) > 1e-4

    if active:
        with self._lock:
            current_mode = self._mode

        if current_mode != ROS_MODE:
            print(f"ROS /cmd_vel start: v={v_cmd:.3f}, omega={omega_cmd:.3f}")
            self._start_drive(ROS_MODE)
        else:
            with self._mj_lock:
                sim_t = float(self.data.time)
            with self._lock:
                self._drive_start_t = sim_t
    else:
        print("ROS /cmd_vel stop")
        with self._lock:
            self._mode = self.MODE_NONE


def _ros_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)

    self._ros_enabled = False
    self._ros_node = None

    self._ros_pub_clock = None
    self._ros_pub_terrain = None
    self._ros_pub_slip = None
    self._ros_pub_odom = None
    self._ros_pub_imu = None
    self._ros_tf_broadcaster = None
    self._ros_sub_cmd_vel = None
    self._ros_spin_thread = None

    self._ros_v_cmd = 0.0
    self._ros_omega_cmd = 0.0

    self._ros_last_time = None
    self._ros_last_pos = None
    self._ros_last_yaw = None

    try:
        if not rclpy.ok():
            rclpy.init(args=None)

        self._ros_node = rclpy.create_node("skid_mujoco_node")

        self._ros_pub_clock = self._ros_node.create_publisher(Clock, "/clock", 10)
        self._ros_pub_terrain = self._ros_node.create_publisher(Float32MultiArray, "/terrain_params", 10)
        self._ros_pub_slip = self._ros_node.create_publisher(Float32MultiArray, "/slip_debug", 10)
        self._ros_pub_odom = self._ros_node.create_publisher(Odometry, "/odom", 10)
        self._ros_pub_imu = self._ros_node.create_publisher(Imu, "/imu", 10)

        self._ros_tf_broadcaster = TransformBroadcaster(self._ros_node)

        self._ros_sub_cmd_vel = self._ros_node.create_subscription(
            Twist,
            "/cmd_vel",
            lambda msg: _ros_cmd_vel_callback(self, msg),
            10
        )

        self._ros_spin_thread = threading.Thread(
            target=rclpy.spin,
            args=(self._ros_node,),
            daemon=True
        )
        self._ros_spin_thread.start()

        self._ros_enabled = True
        print("ROS 2 bridge aktiven:")
        print(" subscribe: /cmd_vel")
        print(" publish: /clock")
        print(" publish: /terrain_params")
        print(" publish: /slip_debug")
        print(" publish: /odom")
        print(" publish: /tf")
        print(" publish: /imu")

    except Exception as e:
        self._ros_enabled = False
        print(f"ROS 2 bridge se ni zagnal: {e}")


def _ros_left_right_command_to_cmd_vel(self, left_cmd, right_cmd):
    if getattr(self, "_mode", None) == ROS_MODE:
        v_cmd = float(getattr(self, "_ros_v_cmd", 0.0))
        omega_cmd = float(getattr(self, "_ros_omega_cmd", 0.0))

        b = max(0.05, float(self._track_width_m))
        v_left = v_cmd - 0.5 * b * omega_cmd
        v_right = v_cmd + 0.5 * b * omega_cmd

        return {
            "v_cmd": v_cmd,
            "omega_cmd": omega_cmd,
            "v_left_cmd_mps": v_left,
            "v_right_cmd_mps": v_right,
        }

    return _original_lr_to_cmd_vel(self, left_cmd, right_cmd)


def _quat_to_yaw(qw, qx, qy, qz):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _make_stamp(sim_time):
    sec = int(sim_time)
    nanosec = int((sim_time - sec) * 1e9)
    return sec, nanosec


def _ros_publish_basic(self, drive_info):
    if not getattr(self, "_ros_enabled", False):
        return

    try:
        sim_time = float(self.data.time)
        sec, nanosec = _make_stamp(sim_time)

        body = self.data.body("sasija")

        pos = body.xpos
        x = float(pos[0])
        y = float(pos[1])
        z = float(pos[2])

        # MuJoCo quaternion order: w, x, y, z
        q = body.xquat
        qw = float(q[0])
        qx = float(q[1])
        qy = float(q[2])
        qz = float(q[3])

        yaw = _quat_to_yaw(qw, qx, qy, qz)

        vx = 0.0
        vy = 0.0
        vz = 0.0
        wz = 0.0

        if self._ros_last_time is not None and self._ros_last_pos is not None:
            dt = sim_time - self._ros_last_time
            if dt > 1e-9:
                vx = (x - self._ros_last_pos[0]) / dt
                vy = (y - self._ros_last_pos[1]) / dt
                vz = (z - self._ros_last_pos[2]) / dt

                dyaw = yaw - self._ros_last_yaw
                while dyaw > math.pi:
                    dyaw -= 2.0 * math.pi
                while dyaw < -math.pi:
                    dyaw += 2.0 * math.pi

                wz = dyaw / dt

        self._ros_last_time = sim_time
        self._ros_last_pos = (x, y, z)
        self._ros_last_yaw = yaw

        # /clock
        clock_msg = Clock()
        clock_msg.clock.sec = sec
        clock_msg.clock.nanosec = nanosec
        self._ros_pub_clock.publish(clock_msg)

        # /terrain_params
        mu, lam, mat_id, mat_name = self.terrain_params(x, y)

        terrain_msg = Float32MultiArray()
        terrain_msg.data = [float(mu), float(lam), float(mat_id)]
        self._ros_pub_terrain.publish(terrain_msg)

        # /slip_debug
        slip_msg = Float32MultiArray()
        if drive_info is not None:
            sigma_v = float(drive_info.get("sigma_v_eff", 0.0))
            sigma_omega = float(drive_info.get("sigma_omega_eff", 0.0))
        else:
            sigma_v = 0.0
            sigma_omega = 0.0

        slip_msg.data = [sigma_v, sigma_omega]
        self._ros_pub_slip.publish(slip_msg)

        # /odom
        odom = Odometry()
        odom.header.stamp.sec = sec
        odom.header.stamp.nanosec = nanosec
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"

        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = z

        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.linear.z = vz
        odom.twist.twist.angular.z = wz

        self._ros_pub_odom.publish(odom)

        # /tf: odom -> base_link
        tf_msg = TransformStamped()
        tf_msg.header.stamp.sec = sec
        tf_msg.header.stamp.nanosec = nanosec
        tf_msg.header.frame_id = "odom"
        tf_msg.child_frame_id = "base_link"

        tf_msg.transform.translation.x = x
        tf_msg.transform.translation.y = y
        tf_msg.transform.translation.z = z

        tf_msg.transform.rotation.x = qx
        tf_msg.transform.rotation.y = qy
        tf_msg.transform.rotation.z = qz
        tf_msg.transform.rotation.w = qw

        self._ros_tf_broadcaster.sendTransform(tf_msg)

        # /imu
        imu = Imu()
        imu.header.stamp.sec = sec
        imu.header.stamp.nanosec = nanosec
        imu.header.frame_id = "base_link"

        imu.orientation.x = qx
        imu.orientation.y = qy
        imu.orientation.z = qz
        imu.orientation.w = qw

        imu.angular_velocity.z = wz

        imu.linear_acceleration.x = 0.0
        imu.linear_acceleration.y = 0.0
        imu.linear_acceleration.z = 0.0

        self._ros_pub_imu.publish(imu)

    except Exception as e:
        print(f"Napaka pri ROS publish: {e}")


def _sample_energy_with_ros(self, mode, mat_name, drive_info, dt):
    result = _original_sample_energy(self, mode, mat_name, drive_info, dt)
    _ros_publish_basic(self, drive_info)
    return result


def _run_with_ros_shutdown(self):
    try:
        return _original_run(self)
    finally:
        try:
            if getattr(self, "_ros_node", None) is not None:
                self._ros_node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


base_sim.TankSimulator.__init__ = _ros_init
base_sim.TankSimulator._sample_energy_locked = _sample_energy_with_ros
base_sim.TankSimulator._left_right_command_to_cmd_vel = _ros_left_right_command_to_cmd_vel
base_sim.TankSimulator.run = _run_with_ros_shutdown


if __name__ == "__main__":
    base = Path(__file__).resolve().parent

    xml_path = base / "E1_asfalt.xml"
    #if not xml_path.exists():
        #xml_path = base / "Neraven_teren.xml"

    sim = base_sim.TankSimulator(str(xml_path))
    sim.run()
