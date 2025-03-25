#include <ros/ros.h>
#include <sensor_msgs/Joy.h>
#include <geometry_msgs/Twist.h>

class TeleopRobot
{
public:
    TeleopRobot()
    {
        nh_.param<int>("axis_linear_x", axis_lin_x, 1);
        nh_.param<int>("axis_linear_y", axis_lin_y, 0);
        nh_.param<int>("axis_angular", axis_ang, 3);
        nh_.param<double>("vel_linear", vlinear, 0.6);
        nh_.param<double>("vel_angular", vangular, 0.5);
        nh_.param<int>("button", ton, 4);

        pub_ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 1);
        sub_ = nh_.subscribe<sensor_msgs::Joy>("joy", 10, &TeleopRobot::joyCallback, this);
    }

private:
    void joyCallback(const sensor_msgs::Joy::ConstPtr& joy)
    {
        geometry_msgs::Twist cmd_vel;
        if (joy->buttons[ton])
        {
            cmd_vel.linear.x = joy->axes[axis_lin_x] * vlinear;
            cmd_vel.linear.y = joy->axes[axis_lin_y] * vlinear;
            cmd_vel.angular.z = joy->axes[axis_ang] * vangular;
            pub_.publish(cmd_vel);
        }
    }

    ros::NodeHandle nh_;
    ros::Publisher pub_;
    ros::Subscriber sub_;
    int axis_ang, axis_lin_x, axis_lin_y, ton;
    double vlinear, vangular;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "joy_control");
    TeleopRobot teleop_robot;
    ros::spin();
    return 0;
}
