#include "detection.cpp"

//PCL inclusions
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/vfh.h>

#include <string>
#include <boost/filesystem.hpp>
#include <flann/flann.h>
// #include <flann/io/hdf5.h>
#include <fstream>
#include <stdlib.h>


using namespace std;
using namespace pcl;

struct ObjectModelInfo{
	string name;
	string metainfo;
};

struct Hypothesis{
	int model; //index for RecognitionResults.models
	float distance;
};

struct DetectedObject{
	PointCloud<PointT> pointCloud;
	vector<Hypothesis> hypothesis;
};

struct RecognitionResults{
	boost::shared_ptr<vector<ObjectModelInfo> > models;
	boost::shared_ptr<vector<DetectedObject> > detectedObjects;
};

typedef std::pair<std::string, std::vector<float> > vfh_model;

class ObjectModel{
private:
	Object::Ptr object;
	string name;
	string metainfo;
	Eigen::Matrix4d pose;
	PointCloud<VFHSignature308>::Ptr histogram;

public:
	// typedef boost::shared_ptr<ObjectModel> Ptr;
	// typedef boost::shared_ptr<vector<ObjectModel::Ptr> > Collection;

	ObjectModel(){ }

	// object from clusters
	ObjectModel(Object::Ptr o){ 

		if(o != 0){
			object = o;
			histogram = extractFeatures();
			name = generateName();
		}else{
			cout << "No object provided!" << endl;
		}

	}

	// object from cad
	ObjectModel(Object::Ptr o, string n){
		object = o;
		histogram = extractFeatures();
		name = n;

	}

	string getName(){ return name;}

	Object::Ptr getObject(){ return object;}

	PointCloud<VFHSignature308>::Ptr extractFeatures(){

		PointCloud<Normal>::Ptr normals(new PointCloud<Normal>());

		//Compute Normals for the single object
        NormalEstimation<PointT, Normal> normal_estimator;
        search::KdTree<PointT>::Ptr treeNormal(new search::KdTree<PointT>());
        normal_estimator.setInputCloud(object->getPointCloud());
        normal_estimator.setSearchMethod(treeNormal);
        normal_estimator.setKSearch(10);
        normal_estimator.compute(*normals);


		//Create the VFH estimation object
		VFHEstimation<PointT, Normal, VFHSignature308> vfh;
		search::KdTree<PointT>::Ptr treeVfh(new search::KdTree<PointT>());
		vfh.setInputCloud(object->getPointCloud());
		vfh.setInputNormals(normals);
		vfh.setSearchMethod(treeVfh);

		//Output datasets
		PointCloud<VFHSignature308>::Ptr vfhs(new PointCloud<VFHSignature308>());
		vfh.compute(*vfhs);

		return vfhs;
	}

	string generateName(){

		int id = rand() % 100;
		stringstream n;
		n << id << "_cluster";
		string name = n.str();		

		return name;
	}

	
};

class ObjectDB{

private:
	string id;

public:

	ObjectDB(){ }

	ObjectDB(string i, vector<ObjectModel> c){
		id = i;
		objects = c;
	}

	void loadDB(string name){

	}

	void writeDB(){

	}


private:

	vector<ObjectModel> objects;

};

class Recognition{
private:
	vector<ObjectModel> detected_objects;
	ObjectDB reference_db;

public:
	Recognition(){}

	Recognition(vector<ObjectModel> objects, string db_name){
		detected_objects = objects;
		ObjectDB reference_db = ObjectDB(db_name,objects);
	}

	void buildIndex(){

	}

	vector<PointCloud<VFHSignature308>::Ptr> extractFeaturesDetectedObjs(){
		vector<PointCloud<VFHSignature308>::Ptr> histograms;
		for(vector<ObjectModel>::iterator object = detected_objects.begin(); object != detected_objects.end(); ++object){
			ObjectModel model = *object;
			histograms.push_back(model.extractFeatures());
		}

		return histograms;
	}

	void nearestKSearch (flann::Index<flann::ChiSquareDistance<float> > &index, const vfh_model &model, 
                			int k, flann::Matrix<int> &indices, flann::Matrix<float> &distances){

}

};