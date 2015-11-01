#include <ros/ros.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <flann/flann.h>
// #include <flann/io/hdf5.h>
#include <pcl/filters/passthrough.h>


// My inclusion
#include "recognition.cpp"

ros::Publisher pub;

void 
cloud_cb (const sensor_msgs::PointCloud2ConstPtr& input)
{
  // Container for original & filtered data
  pcl::PCLPointCloud2* cloud = new pcl::PCLPointCloud2; 
  pcl::PCLPointCloud2ConstPtr cloudPtr(cloud);
  pcl::PCLPointCloud2 result;

  // Convert to PCL data type
  pcl_conversions::toPCL(*input, *cloud);

  // // Convert the sensor_msgs/PointCloud2 data to pcl/PointCloud
  // pcl::PointCloud<pcl::PointXYZ> cloud;
  // pcl::fromROSMsg (*input, cloud);

  //Create the recognition algorithm
  Recognition rec(cloudPtr);

  //Perform the recognition
  // cloud_filtered = rec.planeSegmentation();
  // cloud_filtered = rec.Voxelize();
  result = rec.run();


  // Convert to ROS data type
  sensor_msgs::PointCloud2 output;
  pcl_conversions::moveFromPCL(result, output);

  // Publish the data
  pub.publish (output);
}

int
main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "object_recognition_module");
  ros::NodeHandle nh;

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2> ("input", 1, cloud_cb);

  // Create a ROS publisher for the output point cloud
  pub = nh.advertise<sensor_msgs::PointCloud2> ("output", 1);

  // Spin
  ros::spin ();
}