#include <ros/ros.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
// #include <flann/flann.h>
// #include <flann/io/hdf5.h>
#include <pcl/filters/passthrough.h>


// My inclusion
// #include "detection.cpp"
#include "recognition.cpp"

//Filesystem
#include <dirent.h>
#include <iostream>

namespace fs = boost::filesystem;

ros::Publisher pub;

boost::mutex fb_mutex; 

void 
cloud_cb (const sensor_msgs::PointCloud2ConstPtr& input)
{
  // // Container for original & filtered data
  // PCLPointCloud2* cloud = new PCLPointCloud2; 
  // PCLPointCloud2ConstPtr cloudPtr(cloud);
  PCLPointCloud2 result;

  // // Convert to PCL data type
  // pcl_conversions::toPCL(*input, *cloud);

  // Convert the sensor_msgs/PointCloud2 data to pcl/PointCloud
  PointCloud<PointT> cloud;
  fromROSMsg (*input, cloud);

   // flann::Index<flann::ChiSquareDistance<float> > index();

  //Create the detection algorithm
  Detection det(cloud.makeShared());

  //Detects the objects
  {

    boost::mutex::scoped_lock(db_mutex);

    Object::Collection objects = det.run();
    vector<ObjectModel> models;
    for(vector<Object::Ptr>::iterator obj = objects->begin(); obj != objects->end(); ++obj){
      ObjectModel model(*obj);
      models.push_back(model);
    }

    Recognition rec(models,"reference_objects");
    
  }

  // Convert to ROS data type
  sensor_msgs::PointCloud2 output;
  pcl_conversions::moveFromPCL(result, output);

  // Publish the data
  pub.publish (output);
}

void loadModels(string models_position){
  vector<string> files = vector<string>();
  vector<ObjectModel> reference_objects;

  cout << "Writing the DB for the CAD models..." << endl;

  // Get the files name
  DIR *dp;
  struct dirent *dirp;
  if((dp  = opendir(models_position.c_str())) == NULL) {
    cout << "Error(" << errno << ") opening " << models_position << endl;
  }

  while ((dirp = readdir(dp)) != NULL) {
    if(dirp->d_type == DT_DIR) continue;
    files.push_back(string(dirp->d_name));
  }
  closedir(dp);

  // Load the PointCloud and Creat the ObjectModel
  cout << "Models found: " << endl;
  for(vector<string>::iterator m_name = files.begin(); m_name != files.end(); ++m_name){

    string name = *m_name;

    cout << "- " << name << endl;

    PointCloud<PointT>::Ptr cloud(new PointCloud<PointT>);
    string complete_name = models_position + "/" + name;
    if(io::loadPCDFile<PointT>(complete_name, *cloud) == -1){
      cout << "Couldn't read file " << complete_name << endl;
    }

    Object::Ptr obj = Object::createFromCAD(cloud);
    ObjectModel model(obj,name);
    reference_objects.push_back(model);
  }

  ObjectDB db("reference_objects",reference_objects);
  db.writeDB();
  cout << "...done! " << endl;
  sleep(2);

}

int main (int argc, char** argv){

  // Initialize ROS
  ros::init (argc, argv, "object_recognition_module");
  ros::NodeHandle nh;

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2> ("input", 1, cloud_cb);

  //Create the DB of trained ObjectModels (CAD)
  loadModels("/home/valerio/catkin_ws/src/obj_recognition/model");


  // Create a ROS publisher for the output point cloud
  pub = nh.advertise<sensor_msgs::PointCloud2> ("output", 1);

  // Spin
  ros::spin ();
}