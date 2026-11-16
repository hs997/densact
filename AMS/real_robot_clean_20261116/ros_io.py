"""Small ROS1/ROS2 transport; imported only by ROS executables, not core/tests."""
import threading


class ROSBus:
    def __init__(self, version, name):
        self.version = int(version)
        self.keep = []
        if self.version == 1:
            import rospy
            self.ros = rospy
            rospy.init_node(name, anonymous=False, disable_signals=True)
            self.node = None
        else:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            rclpy.init(args=[])
            self.ros = rclpy
            self.node = rclpy.create_node(name)
            self.executor = SingleThreadedExecutor()
            self.executor.add_node(self.node)
            self.thread = threading.Thread(target=self.executor.spin, daemon=True)
            self.thread.start()
        from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
        from std_msgs.msg import Bool, String
        from std_srvs.srv import Trigger
        self.PoseStamped, self.Twist, self.TwistStamped = PoseStamped, Twist, TwistStamped
        self.Bool, self.String, self.Trigger = Bool, String, Trigger

    def now(self):
        return self.ros.Time.now().to_sec() if self.version == 1 else self.node.get_clock().now().nanoseconds/1e9

    def ok(self):
        return not self.ros.is_shutdown() if self.version == 1 else self.ros.ok()

    def stamp(self, message):
        t = message.header.stamp
        return t.to_sec() if self.version == 1 else float(t.sec)+float(t.nanosec)/1e9

    def subscribe(self, topic, kind, callback, sensor=False):
        if self.version == 1:
            sub = self.ros.Subscriber(topic, kind, callback, queue_size=1, tcp_nodelay=True)
        else:
            from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT if sensor else ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.VOLATILE)
            sub = self.node.create_subscription(kind, topic, callback, qos)
        self.keep.append(sub)

    def publisher(self, topic, kind):
        if self.version == 1:
            pub = self.ros.Publisher(topic, kind, queue_size=1)
        else:
            pub = self.node.create_publisher(kind, topic, 1)
        self.keep.append(pub)
        return pub

    def trigger(self, name, callback):
        if self.version == 1:
            from std_srvs.srv import TriggerResponse
            def run(_):
                ok, text = callback()
                return TriggerResponse(success=ok, message=text)
            server = self.ros.Service(name, self.Trigger, run)
        else:
            def run(_, response):
                response.success, response.message = callback()
                return response
            server = self.node.create_service(self.Trigger, name, run)
        self.keep.append(server)

    def candidate_message(self, speed, steering, stamp, identity):
        message = self.TwistStamped()
        if self.version == 1:
            message.header.stamp = self.ros.Time.from_sec(stamp)
        else:
            seconds = int(stamp)
            message.header.stamp.sec = seconds
            message.header.stamp.nanosec = int(round((stamp-seconds)*1e9))
            if message.header.stamp.nanosec >= 1000000000:
                message.header.stamp.sec += 1
                message.header.stamp.nanosec = 0
        message.header.frame_id = identity
        message.twist.linear.x, message.twist.angular.z = float(speed), float(steering)
        return message

    def twist(self, speed=0.0, angular_z=0.0):
        msg = self.Twist()
        msg.linear.x, msg.angular.z = float(speed), float(angular_z)
        return msg

    def close(self):
        if self.version == 1:
            self.ros.signal_shutdown('AMS controller exit')
        else:
            self.executor.shutdown(timeout_sec=1.0)
            self.node.destroy_node()
            self.ros.shutdown()
