#include <ros/ros.h>
#include <math.h>

#include <boost/make_shared.hpp>

// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/common/common_headers.h>
#include <pcl/filters/project_inliers.h>

// segmentation includes
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>

using namespace pcl;
using namespace std;

typedef PointXYZ PointT;

//Table parameters
#define MAX_ANGLE 10 // Angle btw kinect and floor
#define TABLE_MIN_POINTS 50 // Min number of points for a table
#define TABLE_CLUSTER_TOLERANCE 0.05 // Minimum distance btw two points that are not considered to be in the same table cloud (meters)
#define TABLE_THRESHOLD 0.01 // Distance treshold for the SAC Segmentation for table detection
#define TABLE_MAX_ITERATIONS 100 // Maximum number of iterations for table detection
#define TABLE_MIN_WIDTH 0.20 // Minimum width of a table in m (x-axis)
#define TABLE_MIN_DEPTH 0.15 // Minimum depth of a table in m (z-axis)

//Object parameters
#define OBJ_MIN_POINTS 10 // Minimum number of points of a detected object
#define OBJ_MIN_HEIGHT 0.10 // Minimum height of an object (meters)
#define OBJ_CLUSTER_TOLERANCE 0.05 // Minimum distance between two points that are not considered to be in the same object cloud (meters)
#define OBJ_MIN_DISTANCE_TO_TABLE 0.01 // Minimum distance from an object to a table (meters)
#define OBJ_MAX_DISTANCE_TO_TABLE 1.00 // Maximum distance from an object to a table (meters)

//Filtering parameters
#define DOWNSAMPLIG_RESOLUTION 0.01 // Minimum distance between any two points
#define X_MIN -1
#define X_MAX 1
#define Y_MIN -1
#define Y_MAX 1
#define Z_MIN 1
#define Z_MAX 2.5

class Scene {
    public:
        typedef boost::shared_ptr<Scene> Ptr;

        Scene(){}
        PointCloud<PointT>::ConstPtr getFullPointCloud() { return cloud; }
        PointCloud<PointT>::ConstPtr getDownsampledPointCloud() { return downsampledCloud; }
        PointCloud<PointNormal>::ConstPtr getNormals() { return normals; }

        // static Scene::Ptr fromPointCloud(PointCloud<PointT>::ConstPtr &cloud, ConfigProvider::Ptr config);
        Scene::Ptr createSceneObj(PointCloud<PointT>::Ptr originalCloud){
            // PointCloud<PointT>::Ptr originalCloud(new PointCloud<PointT>);
            // fromPCLPointCloud2(*source_cloud,*originalCloud);

            //Build the result
            Scene::Ptr scene(new Scene());
            scene->cloud = originalCloud;
            scene->downsampledCloud = boost::make_shared<PointCloud<PointT> >();
            scene->normals = boost::make_shared<PointCloud<PointNormal> >();

            //Crop the PointCloud
            PassThrough<PointT> filter;
            PointCloud<PointT>::Ptr cropCloud(new PointCloud<PointT>());
            filter.setInputCloud(scene->cloud);
            filter.setFilterFieldName("x");
            filter.setFilterLimits(X_MIN,X_MAX);
            filter.setFilterFieldName("y");
            filter.setFilterLimits(Y_MIN,Y_MAX);
            filter.setFilterFieldName("z");
            filter.setFilterLimits(Z_MIN,Z_MAX);
            filter.filter(*cropCloud);
            scene->cloud = cropCloud;

            //Downsample Data
            VoxelGrid<PointT> downsampler;
            downsampler.setInputCloud(scene->cloud);
            downsampler.setLeafSize(DOWNSAMPLIG_RESOLUTION, DOWNSAMPLIG_RESOLUTION, DOWNSAMPLIG_RESOLUTION);
            downsampler.filter(*(scene->downsampledCloud));

            //Compute Normals
            NormalEstimation<PointT, PointNormal> normal_estimator;
            search::KdTree<PointT>::Ptr tree(new search::KdTree<PointT>());
            normal_estimator.setInputCloud(scene->downsampledCloud);
            normal_estimator.setSearchMethod(tree);
            normal_estimator.setKSearch(10);
            normal_estimator.compute(*(scene->normals));

            return scene;
        }

    private:
        PointCloud<PointT>::ConstPtr cloud;
        PointCloud<PointT>::Ptr downsampledCloud;
        PointCloud<PointNormal>::Ptr normals;


};

class Table {
    public:
        typedef boost::shared_ptr<Table> Ptr;
        typedef boost::shared_ptr<vector<Table::Ptr> > Collection;
        
        PointCloud<PointT>::ConstPtr getConvexHull() { return convexHull; }
        ModelCoefficientsConstPtr getModelCoefficients() { return modelCoefficients; }
        PointT getMinDimensions() { return minDimensions; }
        PointT getMaxDimensions() { return maxDimensions; }
        double getWidth() { return maxDimensions.x - minDimensions.x; }
        double getDepth() { return maxDimensions.z - minDimensions.z; }

        static Table::Ptr fromConvexHull(PointCloud<PointT>::ConstPtr hull, ModelCoefficients::ConstPtr modelCoefficients){

        	Table::Ptr table(new Table());
        	table->convexHull = hull;
        	table->modelCoefficients = modelCoefficients;
        	getMinMax3D(*hull, table->minDimensions, table->maxDimensions);

        	return table;
        }

    private:
        PointCloud<PointT>::ConstPtr convexHull;
        ModelCoefficients::ConstPtr modelCoefficients;
        PointT minDimensions;
        PointT maxDimensions;
        
        Table(){}
};

class Object {
    public:
        typedef boost::shared_ptr<Object> Ptr;
        typedef boost::shared_ptr<vector<Object::Ptr> > Collection;

        Object(){}
        PointCloud<PointT>::ConstPtr getPointCloud() { return pointCloud; }
        PointCloud<PointT>::ConstPtr getBaseConvexHull() { return baseConvexHull; }
        PointT getMinDimensions() { return minDimensions; }
        PointT getMaxDimensions() { return maxDimensions; }
        double getWidth() { return maxDimensions.x - minDimensions.x; }
        double getHeight() { return maxDimensions.y - minDimensions.y; }
        int getPointCount() { return getPointCloud()->points.size(); }

        static Object::Ptr create(PointCloud<PointT>::ConstPtr pointCloud, PointCloud<PointT>::ConstPtr baseConvexHull){
            Object::Ptr object(new Object());
            object->pointCloud = pointCloud;
            object->baseConvexHull = baseConvexHull;
            getMinMax3D(*pointCloud, object->minDimensions, object->maxDimensions);

            return object;
        }

        static Object::Ptr createFromCAD(PointCloud<PointT>::Ptr pointCloud){
            Object::Ptr object(new Object());
            object->pointCloud = pointCloud;
            getMinMax3D(*pointCloud, object->minDimensions, object->maxDimensions);

            return object;
        }

    private:
        PointCloud<PointT>::ConstPtr pointCloud;
        PointCloud<PointT>::ConstPtr baseConvexHull;
        PointT minDimensions;
        PointT maxDimensions;

};

class Detection{

protected:

	Scene::Ptr scene;

	Table::Collection detectedTables;
	Object::Collection detectedObjects;

public:

	Detection(PointCloud<PointT>::Ptr c){
        Scene app;
		scene = app.createSceneObj(c);
	}

	Detection(){

	}

	~Detection(){

	}


	Table::Collection detectTables(){

		Table::Collection foundTables(new vector<Table::Ptr>());

		PointCloud<PointNormal>::ConstPtr normals = scene->getNormals();
		PointCloud<PointT>::ConstPtr cloud = scene->getDownsampledPointCloud();

		//Filter points by normals
		PointCloud<PointT>::Ptr candidates(new PointCloud<PointT>);
		double maxAngle = cos(M_PI * MAX_ANGLE / 180.0);
		for(int i = 0; i < normals->points.size(); ++i){
			if(normals->points[i].normal_y >= maxAngle || normals->points[i].normal_y <= -maxAngle){
				candidates->points.push_back(cloud->points[i]);
			}
		}

		//Check if enough points
		if(candidates->points.size() < TABLE_MIN_POINTS){
			cout << "Not enough points to estimate a table! " << endl;
		}

		//Cluster the candidates
		vector<PointIndices> tableClusters;
		EuclideanClusterExtraction<PointT> clusterExtractor;
		search::KdTree<PointT>::Ptr kdtree(new search::KdTree<PointT>);
		kdtree->setInputCloud(candidates);
        clusterExtractor.setInputCloud(candidates);
        clusterExtractor.setSearchMethod(kdtree);
        clusterExtractor.setMinClusterSize(TABLE_MIN_POINTS);
        clusterExtractor.setClusterTolerance(TABLE_CLUSTER_TOLERANCE);
        clusterExtractor.extract(tableClusters);

        // set RANSAC to find plane model
        SACSegmentation<PointT> segmentation;
        segmentation.setModelType(SACMODEL_PLANE);
        segmentation.setMethodType(SAC_RANSAC);
        segmentation.setProbability(0.99);
        segmentation.setMaxIterations(TABLE_MAX_ITERATIONS);
        segmentation.setDistanceThreshold(TABLE_THRESHOLD);
        segmentation.setOptimizeCoefficients(false);

        //FOR EACH CLUSTER
        for(vector<PointIndices>::iterator cluster = tableClusters.begin(); cluster != tableClusters.end(); ++cluster){

        	// Find a plane model
        	ModelCoefficients::Ptr planeCoefficients(new ModelCoefficients());
        	PointIndices::Ptr tableIndices(new PointIndices());
        	segmentation.setInputCloud(candidates);
        	segmentation.setIndices(boost::make_shared<PointIndices>(*cluster));
        	segmentation.segment(*tableIndices, *planeCoefficients);

        	// Project the table points on the plane model
        	PointCloud<PointT>::Ptr projectedTable(new PointCloud<PointT>());
            ProjectInliers<PointT> projector;
            projector.setInputCloud(candidates);
            projector.setIndices(tableIndices);
            projector.setModelCoefficients(planeCoefficients);
            projector.setModelType(SACMODEL_PLANE);
            projector.filter(*projectedTable);

            // Calculates the 2-D convex hull
            PointCloud<PointT>::Ptr tableHull(new PointCloud<PointT>());
            ConvexHull<PointT> convex_hull;
            convex_hull.setInputCloud(projectedTable);
            convex_hull.reconstruct(*tableHull);

            // check for minimal width and depth
            Table::Ptr table = Table::fromConvexHull(tableHull, planeCoefficients);
            if (table->getWidth() < TABLE_MIN_WIDTH) continue;
            if (table->getDepth() < TABLE_MIN_DEPTH) continue;
            foundTables->push_back(table);

        }

     //    if(foundTables->size() == 0) cout << "No table found! " << endl;
     //    else{
     //    	cout << "Number of tables found " << foundTables->size() << endl;
     //    	PointCloud<PointT> tableHull = *foundTables->at(0)->getConvexHull();
     //    	toPCLPointCloud2(tableHull,result);
    	// }

        return foundTables;

	}

	Object::Collection detectObjects(Table::Collection tables){
        Object::Collection foundObjects(new vector<Object::Ptr>());

        // foreach table
        for (vector<Table::Ptr>::iterator table = tables->begin(); table != tables->end(); ++table) {

            // extract all points above the table
            PointIndices::Ptr indicesAllObjects(new PointIndices());
            ExtractPolygonalPrismData<PointT> objectExtractor;
            objectExtractor.setInputCloud(scene->getDownsampledPointCloud());
            objectExtractor.setInputPlanarHull((*table)->getConvexHull());
            objectExtractor.setHeightLimits(OBJ_MIN_DISTANCE_TO_TABLE, OBJ_MAX_DISTANCE_TO_TABLE);
            objectExtractor.segment(*indicesAllObjects);

            if (indicesAllObjects->indices.size() < 1) continue;

            // project all objects to the table plane
            PointCloud<PointT>::Ptr cloudAllObjectsProjected(new PointCloud<PointT>());
            ProjectInliers<PointT> projectionTable;
            projectionTable.setInputCloud(scene->getDownsampledPointCloud());
            projectionTable.setIndices(indicesAllObjects);
            projectionTable.setModelType(SACMODEL_PLANE);
            projectionTable.setModelCoefficients((*table)->getModelCoefficients());
            projectionTable.filter(*cloudAllObjectsProjected);

            // split the object cloud into clusters (object candidates)
            vector<PointIndices> objectClusters;
            EuclideanClusterExtraction<PointT> objectClusterer;
            search::KdTree<PointT>::Ptr treeObjects(new search::KdTree<PointT>);
            treeObjects->setInputCloud(cloudAllObjectsProjected);
            objectClusterer.setInputCloud(cloudAllObjectsProjected);
            objectClusterer.setSearchMethod(treeObjects);
            objectClusterer.setMinClusterSize(OBJ_MIN_POINTS);
            objectClusterer.setClusterTolerance(OBJ_CLUSTER_TOLERANCE);
            objectClusterer.extract(objectClusters);

            // for each object cluster...
            for (vector<PointIndices>::iterator objectCluster = objectClusters.begin(); objectCluster != objectClusters.end(); ++objectCluster) {
                PointIndices::Ptr indicesProjectedObject(new PointIndices(*objectCluster));

                // calculate the convex hull of the (projected) object candidate
                PointCloud<PointT>::Ptr hullProjectedObject(new PointCloud<PointT>());
                ConvexHull<PointT> convexHullCalculator;
                convexHullCalculator.setInputCloud(cloudAllObjectsProjected);
                convexHullCalculator.setIndices(indicesProjectedObject);
                convexHullCalculator.reconstruct(*hullProjectedObject);

                // get all points over the projected object (using original cloud, not the downsampled one!)
                PointIndices::Ptr indicesObject(new PointIndices());
                objectExtractor.setInputCloud(scene->getFullPointCloud());
                objectExtractor.setInputPlanarHull(hullProjectedObject);
                objectExtractor.segment(*indicesObject);

                // convert indices to the final point cloud
                PointCloud<PointT>::Ptr cloudObject(new PointCloud<PointT>());
                ExtractIndices<PointT> extractor;
                extractor.setInputCloud(scene->getFullPointCloud());
                extractor.setIndices(indicesObject);
                extractor.filter(*cloudObject);

                // create Object object
                Object::Ptr object = Object::create(cloudObject, hullProjectedObject);
                if (object->getHeight() < OBJ_MIN_HEIGHT) continue;
                if (object->getPointCount() < OBJ_MIN_POINTS) continue;

                // is the object flying?
                if (object->getMinDimensions().y - (*table)->getMinDimensions().y > 2 * OBJ_MIN_DISTANCE_TO_TABLE) continue;

                // save detected object
                foundObjects->push_back(object);
            }
        }

        return foundObjects;
    }

    Object::Collection run(){

        Table::Collection detectedTables = detectTables();
        Object::Collection detectedObjects = detectObjects(detectedTables);

        // detectedTables = detectedTables;
        // detectedObjects = detectedObjects;


        return detectedObjects;
    }

    // PointCloud<PointNormal>::Ptr getNormals(){

    //  PointCloud<PointNormal>::Ptr normals(new PointCloud<PointNormal>);

    //  PointCloud<PointT>::Ptr scene(new PointCloud<PointT>);
    //  fromPCLPointCloud2(*cloud,*scene);

    //  NormalEstimation<PointT, PointNormal> normal_estimator;
    //  search::KdTree<PointT>::Ptr treeNormals(new search::KdTree<PointT>());
    //  normal_estimator.setInputCloud(scene);
    //  normal_estimator.setSearchMethod(treeNormals);
    //  normal_estimator.setKSearch(10);
    //  normal_estimator.compute(*normals);

    //  return normals;
    // }

	// PCLPointCloud2 planeSegmentation(){

	// 	PCLPointCloud2 result;
	// 	PointCloud<PointT>::Ptr seg_cloud(new PointCloud<PointT>);
	// 	fromPCLPointCloud2(*cloud,*seg_cloud);

	// 	PointCloud<PointT>::Ptr plane(new PointCloud<PointT>);
	// 	PointCloud<PointT>::Ptr convex_hull(new PointCloud<PointT>);
	// 	PointCloud<PointT>::Ptr objects(new PointCloud<PointT>);


	// 	SACSegmentation<PointT> segmentation;
	// 	segmentation.setInputCloud(seg_cloud);
	// 	segmentation.setModelType(SACMODEL_PLANE);
	// 	segmentation.setMethodType(SAC_RANSAC);
	// 	segmentation.setDistanceThreshold(0.01);
	// 	segmentation.setOptimizeCoefficients(true);

	// 	PointIndices::Ptr planeIndices(new PointIndices);
	// 	ModelCoefficients::Ptr coefficients(new ModelCoefficients);
	// 	segmentation.segment(*planeIndices, *coefficients);

	// 	if(planeIndices->indices.size() == 0){
	// 		cout << "Could not find a plane in the scene." << endl;
	// 		// exit(0); //must throw an error
	// 	}
	// 	else{

	// 		//Extract the plane
	// 		ExtractIndices<PointT> extract;
	// 		extract.setInputCloud(seg_cloud);
	// 		extract.setIndices(planeIndices);
	// 		extract.filter(*plane);

	// 		//Retrieve the Convex Hull
	// 		ConvexHull<PointT> hull;
	// 		hull.setInputCloud(plane);
	// 		hull.setDimension(2);
	// 		hull.reconstruct(*convex_hull);

	// 		//Prim object
	// 		ExtractPolygonalPrismData<PointT> prism;
	// 		prism.setInputCloud(seg_cloud);
	// 		prism.setInputPlanarHull(convex_hull);

	// 		//params: min z value, max z value (depending on the objects)
	// 		prism.setHeightLimits(0.0f,0.1f);
	// 		PointIndices::Ptr objectIndices(new PointIndices);

	// 		prism.segment(*objectIndices);

	// 		extract.setIndices(objectIndices);
	// 		extract.filter(*objects);

	// 	}

	// 	toPCLPointCloud2(*objects,result);

	// 	return result;
	// }



	// PCLPointCloud2 Voxelize(){
	// 	PCLPointCloud2 cloud_filtered;

	// 	VoxelGrid<PCLPointCloud2> sor;
	// 	sor.setInputCloud (cloud);
	// 	sor.setLeafSize (0.01, 0.01, 0.01);
	// 	sor.filter (cloud_filtered);

	// 	return cloud_filtered;
	// }

	// PCLPointCloud2 PassThrough(float xmin, float ymin, float zmin, float max){
		
	// 	PCLPointCloud2 cloud_filtered;

	// 	PassThrough<PCLPointCloud2> pass;
	// 	pass.setInputCloud (cloud);
	// 	pass.setFilterFieldName("x");
	// 	pass.setFilterLimits(xmin,xmin+max);
	// 	pass.setFilterFieldName("y");
	// 	pass.setFilterLimits(ymin, ymin+max);
	// 	pass.setFilterFieldName("z");
	// 	pass.setFilterLimits(zmin, zmin+max);
	// 	pass.filter(cloud_filtered);

	// 	return cloud_filtered;

	// }

};