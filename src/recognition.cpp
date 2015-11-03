#include "detection.cpp"

//PCL inclusions
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/vfh.h>
#include <pcl/console/parse.h>
#include <pcl/console/print.h>

#include <string>
#include <boost/filesystem.hpp>
#include <flann/flann.h>
// #include <flann/io/hdf5.h>
#include <fstream>
#include <stdlib.h>

//XML Writer & Parser
#include "../tinyxml/tinystr.h"
#include "../tinyxml/tinystr.cpp"
#include "../tinyxml/tinyxml.h"
#include "../tinyxml/tinyxml.cpp"
#include "../tinyxml/tinyxmlerror.cpp"

#define HYPOTHESIS_COUNT 5 // the number of object models that fit the detected object best


const string db_path = "/home/valerio/catkin_ws/src/obj_recognition/db/";

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
	ObjectModelInfo info;
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
			info.name = generateName();
		}else{
			cout << "No object provided!" << endl;
		}

	}

	// object from cad
	ObjectModel(Object::Ptr o, string n){
		object = o;
		histogram = extractFeatures();
		info.name = n;

	}

	string getName(){ return info.name;}

	Object::Ptr getObject(){ return object;}

	PointCloud<VFHSignature308>::Ptr getHistogram(){ return histogram; }

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

/**
** ObjectDB: A Class representing the DB with the reference model to match against
**/
class ObjectDB{

private:
	string id;
	vector<ObjectModel> objects;

public:

	ObjectDB(){ }

	ObjectDB(string i, vector<ObjectModel> c){
		id = i;
		objects = c;
	}

	string getId(){ return id; }

	vector<ObjectModel> getObjects(){ return objects; }

	//Load the DB with all the models
	void loadDB(string name){

	}

	//Write the DB to a file
	void writeDB(){

		//Declaration XML Format
		TiXmlDocument doc;
		TiXmlDeclaration* pDecl = new TiXmlDeclaration("1.0", "", "");
		doc.LinkEndChild(pDecl);

		const char* db_name = getId().c_str();

		//Add root node (DB Name)
		TiXmlElement* pRoot = new TiXmlElement(db_name);
		doc.LinkEndChild(pRoot);

		int j = 0;
		for(vector<ObjectModel>::iterator it = objects.begin(); it != objects.end(); ++it, ++j){

			ObjectModel obj = *it;

			// Add child node and data (Object Name)
			TiXmlElement* pElem = new TiXmlElement(obj.getName().c_str());
			pRoot->LinkEndChild(pElem);



			//Object Histogram
			PointCloud<VFHSignature308> hist = *obj.getHistogram();
			PCLPointCloud2 cloud;
			toPCLPointCloud2(hist,cloud);
			int vfh_idx = getFieldIndex(cloud, "vfh");
			vector<PCLPointField> fields;
			getFieldIndex(hist, "vfh", fields);
			vfh_model vfh;
			vfh.second.resize(308);

			const string num = boost::lexical_cast<string>(j);
			string h = "";
			for(size_t i = 0; i < fields[vfh_idx].count; ++i){
				vfh.second[i] = hist.points[0].histogram[i];
				h +=  boost::lexical_cast<string>(vfh.second[i]) + "; ";
			// const string app = "histogram_" + num;
			// const char* name = app.c_str();

			// 	pElem->SetDoubleAttribute(name,double(hist.points[0].histogram[i]));
			}

			//Write the Histogram as text
			TiXmlElement* pSubElem = new TiXmlElement("Histogram"); 
 			pSubElem->LinkEndChild( new TiXmlText(h.c_str()));  
 			pElem->LinkEndChild(pSubElem); 
		}	

		//Save xml file format
		const char* file_name;
		const string xml = ".xml";
		const string result = db_path + db_name + xml;
		file_name = result.c_str();

		doc.SaveFile(file_name);


	}


};

class Recognition{
private:
	vector<ObjectModel> detected_objects;
	vector<vfh_model> detected_objects_vfh;
	// flann::Index<flann::ChiSquareDistance<float> > detected_objects_index;
	ObjectDB reference_db;

public:
	Recognition(){}

	Recognition(vector<ObjectModel> objects, string db_name){
		detected_objects = objects;

		//Load the DB with CAD models for matching
		ObjectDB reference_db;
		reference_db.loadDB(db_name);
	}

	/** \brief Load a set of VFH features that will act as the model (training data)
	  * \param models the resultant vector of histogram models
	  */
	void loadFeaturesModels(){

		vector<vfh_model> models;
		// models.resize(detected_objects.size());

		for(vector<ObjectModel>::iterator it = detected_objects.begin(); it != detected_objects.end(); ++it){
			ObjectModel model = *it;
			vfh_model vfh;
			vfh.second.resize(308);

			PointCloud<VFHSignature308> point = *model.getHistogram();
			PCLPointCloud2 cloud;
			toPCLPointCloud2(point,cloud);
			int vfh_idx = getFieldIndex(cloud, "vfh");
			vector<PCLPointField> fields;
			getFieldIndex(point, "vfh", fields);

			for(size_t i = 0; i < fields[vfh_idx].count; ++i){
				vfh.second[i] = point.points[0].histogram[i];
			}
			vfh.first = model.getName();

			models.push_back(vfh);
		}

		detected_objects_vfh = models;

	}

	flann::Index<flann::ChiSquareDistance<float> > buildIndex(){

		//Load the model histograms
		loadFeaturesModels();
		if(detected_objects_vfh.size() == 0){
			cout << "no models for the index! " << endl;
			// exit(0);
		}

		//Convert data into FLANN format
		flann::Matrix<float> data(new float[detected_objects_vfh.size() * detected_objects_vfh[0].second.size()], detected_objects_vfh.size(), detected_objects_vfh[0].second.size());
		for (size_t i = 0; i < data.rows; ++i)
    		for (size_t j = 0; j < data.cols; ++j)
      			data[i][j] = detected_objects_vfh[i].second[j];

      	// Build the tree index
      	flann::Index<flann::ChiSquareDistance<float> > index(data, flann::LinearIndexParams());
      	index.buildIndex();

      	delete[] data.ptr();

      	return index;
	}

	// vector<PointCloud<VFHSignature308>::Ptr> extractFeaturesDetectedObjs(){
	// 	vector<PointCloud<VFHSignature308>::Ptr> histograms;
	// 	for(vector<ObjectModel>::iterator object = detected_objects.begin(); object != detected_objects.end(); ++object){
	// 		ObjectModel model = *object;
	// 		histograms.push_back(model.extractFeatures());
	// 	}

	// 	return histograms;
	// }

	/** \brief Search for the closest k neighbors
	* \param index the tree
	* \param model the query model (CAD in this case)
	* \param k the number of neighbors to search for
	* \param indices the resultant neighbor indices
	* \param distances the resultant neighbor distances
	*/
	void nearestKSearch (flann::Index<flann::ChiSquareDistance<float> > &index, const vfh_model &model, 
                			int k, flann::Matrix<int> &indices, flann::Matrix<float> &distances){
		//Query point
		flann::Matrix<float> p = flann::Matrix<float>(new float[model.second.size()], 1, model.second.size());
		memcpy (&p.ptr()[0], &model.second[0], p.cols * p.rows * sizeof (float));

		indices = flann::Matrix<int>(new int[k], 1, k);
		distances = flann::Matrix<float>(new float[k], 1, k);
		index.knnSearch(p, indices, distances, k, flann::SearchParams(512));

		delete[] p.ptr();
	}

	void run(){

		cout << "RECOGNITION: " << endl;
		cout << "Writing the flann::Index for the detected objects..." << endl;
		flann::Index<flann::ChiSquareDistance<float> > detected_objects_index = buildIndex();
		cout << ".... DONE!" << endl;

		cout << "loading reference model from DB: " << reference_db.getId() << endl;
		vector<ObjectModel> cad_models = reference_db.getObjects();

		for(vector<ObjectModel>::iterator it = cad_models.begin(); it != cad_models.end(); ++it){

			ObjectModel cad_model = *it;

			vfh_model vfh;
			PointCloud<VFHSignature308> point = *cad_model.getHistogram();
			PCLPointCloud2 cloud;
			toPCLPointCloud2(point,cloud);
			int vfh_idx = getFieldIndex(cloud, "vfh");
			vector<PCLPointField> fields;
			getFieldIndex(point, "vfh", fields);

			for(size_t i = 0; i < fields[vfh_idx].count; ++i){
				vfh.second[i] = point.points[0].histogram[i];
			}
			vfh.first = cad_model.getName();

			flann::Matrix<int> k_indices;
			flann::Matrix<float> k_distances;
			nearestKSearch(detected_objects_index,vfh,HYPOTHESIS_COUNT,k_indices,k_distances);

			// Output the results on screen
			const char* cad_name = cad_model.getName().c_str();
	  		console::print_highlight ("The closest %d neighbors for %s are:\n", HYPOTHESIS_COUNT, cad_name);
	  		for (int i = 0; i < HYPOTHESIS_COUNT; ++i)
	    		console::print_info ("    %d - %s (%d) with a distance of: %f\n",i, detected_objects_vfh.at (k_indices[0][i]).first.c_str (), k_indices[0][i], k_distances[0][i]);
	    }
	}

};