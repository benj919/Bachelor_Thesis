/*=====================================================================

MAVCONN Micro Air Vehicle Flying Robotics Toolkit
Please see our website at <http://MAVCONN.ethz.ch>

(c) 2009, 2010 MAVCONN PROJECT

This file is part of the MAVCONN project

    MAVCONN is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    MAVCONN is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with MAVCONN. If not, see <http://www.gnu.org/licenses/>.

======================================================================*/

/**
 * @file
 *   @brief LCM example
 *
 *   @author Benjamin Flueck <bflueck@student.ethz.ch>
 *
 */

#include <cstdio>
#include <unistd.h>
#include <sigc++-2.0/sigc++/signal.h>
#include <glib-2.0/glib.h>
#include <glibmm-2.4/glibmm.h>
#include <mavlink.h>
#include <mavconn.h>
#include <object_detection/object_detection.h>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <core/MAVConnParamClient.h>
#include <interface/shared_mem/PxSHMImageClient.h>

// Latency Benchmarking
#include <sys/time.h>
#include <time.h>

// Timer for benchmarking
struct timeval tv;

int sysid = 42;
int compid = 112;
bool verbose = false;
bool debug = false;
bool display = false;



int imageCounter = 0;
std::string fileBaseName("frame");
std::string fileExt(".png");

bool quit = false;


MAVConnParamClient* paramClient;
static GString* config_file = g_string_new("../conf/main.cfg");

// for logging
bool logging = false;
std::ofstream log_file;

// Setup for lcm decoupling
PxSHMImageClient ImageClient;
lcm_t* lcm = NULL;
mavlink_message_t imageMessage;
Glib::StaticMutex image_mutex = GLIBMM_STATIC_MUTEX_INIT;;			//mutex controlling the access to the image data
Glib::Cond* image_cond = NULL;
bool newImage = false;


/*
 * Data-structure for one detected person, containing:
 * -an image patch,
 * -the person's position in world-coordinates,
 * -three error-vectors
 * -estimated height of a person
 * -confidence indicator,
 * -an identifier to give the tracker some preliminary information
*/
struct person{
	cv::Mat patch;
	cv::Mat position;
	cv::Mat error_x;
	cv::Mat error_y;
	cv::Mat error_z;
	float height;
	float confidence;
	int id;
	uint64 last_timestamp;
};

// vector to hold the list of persons that are currently tracked in the frames
std::vector<person> persons;


int ident_number=0;
float max_speed = 1;
bool request_redetection = true;
uint64 current_time, last_hog_detection;
uint64 hog_timer = 1000000;
int img_width, img_height;

float current_yaw, last_hog_yaw;
float threshold_yaw = 10. / 180 * 3.141596;

float confidence_threshold = 0.8;

float f;
float mx,my;
float cam_height;
float principle_x, principle_y;

// TODO remove hardcoded pi
float viewing_angle = 60. / 180 * 3.141596;
float max_distance = 10;
int detection_error_x;
int detection_error_y;


cv::HOGDescriptor hog;
cv::Mat Wp;
cv::Mat P2C,C2P,C2B,B2C,W2B,B2W,To2p,pos_pixhawk;


void
signalHandler(int signal)
{
	if (signal == SIGINT)
	{
		fprintf(stderr, "# INFO: Quitting...\n");
		quit = true;
		exit(EXIT_SUCCESS);
	}
}

void print_mat(cv::Mat& mat )
{
    int i, j;
    for( i = 0; i < mat.rows; i++ )
    {
        for( j = 0; j < mat.cols; j++ )
        {
            printf( "%f ", mat.at<float>( i, j ) );
        }
        printf( "\n" );
    }
    printf( "\n" );
}

void update_parameters(){
	max_speed = paramClient->getParamValue("MAX_SPEED");
	viewing_angle = (paramClient->getParamValue("VIEWING_ANGLE"))/ 180 * 3.141596;
	detection_error_x = (int)paramClient->getParamValue("DET_ERROR_X");
	detection_error_y = (int)paramClient->getParamValue("DET_ERROR_Y");
	f = paramClient->getParamValue("CAMERA_F");
	mx = paramClient->getParamValue("CAMERA_MX");
	my = paramClient->getParamValue("CAMERA_MY");
	cam_height = paramClient->getParamValue("CAM_HEIGHT");
	principle_x = paramClient->getParamValue("PRINCIPLE_X");
	principle_y = paramClient->getParamValue("PRINCIPLE_Y");
}

void set_rot_trans_matrices(float pos_x, float pos_y, float pos_z, float ro, float pi, float ya){

	current_yaw = ya;

	pos_pixhawk = (cv::Mat_<float>(4,1) <<
				pos_x,
				pos_y,
				pos_z,
				1
		);

	// pixels to camera
	C2P = (cv::Mat_<float>(4,4) <<
			mx/f, 0,0, principle_x,
			0, my/f,0, principle_y,
			0, 0, 1/f, 0,
			0, 0, 0, 1);
	P2C = C2P.inv();

	//Camera to Body
	C2B = (cv::Mat_<float>(4,4) <<
			0,0,1,0,
			1,0,0,0,
			0,1,0,0,
			0,0,0,1
	);
	B2C = C2B.inv();

	// roll, yaw ,pitch rotations
	cv::Mat Rx = (cv::Mat_<float>(4, 4) <<
			1, 0,		0,		 0,
			0, cos(ro), -sin(ro),0,
			0, sin(ro), cos(ro), 0,
			0, 0,		0,		 1);

	cv::Mat Ry = (cv::Mat_<float>(4, 4) <<
			cos(pi), 0, -sin(pi),0,
			0,		 1,	 0,		 0,
			sin(pi), 0, cos(pi), 0,
			0,		 0, 0,		 1);

	cv::Mat Rz = (cv::Mat_<float>(4, 4) <<
			cos(ya), -sin(ya), 	0, 0,
			sin(ya), cos(ya), 	0, 0,
			0,		 0,			1, 0,
			0,		 0,			0, 1);

	// from world to body frame Rz*Ry*Rx is used -> invert for body to world coords
	W2B = Rz * Ry * Rx;

	B2W = W2B.inv();

	// create translation-matrix origin to pixhawk
	To2p = (cv::Mat_<double>(4,4) <<
			1,0,0,-pos_x,
			0,1,0,-pos_y,
			0,0,1,-pos_z,
			0,0,0,1
	);


}

void pixel_to_world(int pixel_u, int pixel_v, float height, cv::Mat& result){
	// Takes the pixel coordinates u,v and returns the position in 3d space,
	// depending on the height (e.g. zero for ground-plane estimation or known height )
	result = (cv::Mat_<float>(4,1) <<
					pixel_u,
					pixel_v,
					1.,
					1.
					);
	result = (B2W * (C2B * (P2C * result)));
	result = pos_pixhawk + ( ( (height - pos_pixhawk.at<float>(2)) / result.at<float>(2) ) * result);
	result.at<float>(3) = 1;
	//printf("pixel: %d %d to world: %f %f %f\n", pixel_u, pixel_v,result.at<float>(0),result.at<float>(1),result.at<float>(2));
}

void world_to_pixel(cv::Mat& position, int& pixel_x, int& pixel_y){
	// Takes a point in 3D space and returns its pixel-coordinates
	// position of the bottom-center pixel
	cv::Mat bc = /* C2P* */(B2C * (W2B * ( position - pos_pixhawk )));
	bc = (f / (bc.at<float>(2)) * bc);
	bc.at<float>(3) = 1;
	bc = C2P * bc;
	pixel_x = cvRound(bc.at<float>(0));
	pixel_y = cvRound(bc.at<float>(1));
}

float estimate_height(cv::Mat& position, int pixel_u, int pixel_v){
	// estimate the height of a person given the pixel location of the head and the position
	cv::Mat result = (cv::Mat_<float>(4,1)<<
					pixel_u,
					pixel_v,
					1,
					1
					);
	result = (B2W * (C2B * (P2C * result)));
	float t = sqrt( pow( position.at<float>(0) - pos_pixhawk.at<float>(0) , 2)
				   +pow( position.at<float>(1) - pos_pixhawk.at<float>(1) , 2))
			/ sqrt( pow(result.at<float>(0),2) + pow(result.at<float>(1),2));
	float height = pos_pixhawk.at<float>(2) + t * result.at<float>(2);
	return height;
}

bool validate_detected_person(person person){
	// check for the detected position to be "plausible", aka in front of the the camera/vehicle and not too far away,
	// by rotating and translating the position into body coordinates.
	cv::Mat object;
	person.position.copyTo(object);
	object -=  pos_pixhawk;
	object = (W2B * object);
	// get the angle between the "pixhawk to object" -vector and the unit vector in x direction (hardcoded  to the front)
	float dot =  object.at<float>(0);
	float abs = sqrt( pow( object.at<float>(0),2)
					+ pow( object.at<float>(1),2) );
	float angle = acos( dot / abs);
	//printf("validation: angle is%f max%f dist:%f\n",angle, viewing_angle,abs);
	if(object.at<float>(0) < 2.5 || angle > viewing_angle / 2 || abs > max_distance){
		// object is behind the pixhawk or outside the viewing angle or too far away
		return false;
	}
	if(person.height > -1.5 || person.height < -2.5){
		return false;
	}
	return true;
}

void set_detection_errors(person& person, int u, int v, int absolut){
	// update errors, if absolut overwrite else add up. u, v denote the errors in pixel
	int a,b;
	world_to_pixel(person.position,a,b);
	cv::Mat error_x = (cv::Mat_<float>(4,1) <<
			0,
			0,
			0,
			1.
	);
	cv::Mat error_y = (cv::Mat_<float>(4,1) <<
			0,
			0,
			0,
			1.
	);
	cv::Mat error_z = (cv::Mat_<float>(4,1) <<
			0,
			0,
			0,
			0.
	);
	// currently no elevation/height error, therefore 0
	error_z.copyTo(person.error_z);

	// take the larger error
	//float error_factor = 1- ( sqrt(pow(person.position.at<float>(0),2) + pow(person.position.at<float>(1),2)) - 2) / 15;
	float error_factor = 1;
	pixel_to_world(a , cvRound(b - (v)*error_factor), 0, error_x );
	pixel_to_world(((a>img_width/2)? a+u : a-u),b , 0, error_y);
	// orientation doesn't matter, simplified to symmetric error

	error_x -= person.position;
	error_y -= person.position;

	if(absolut == 1){
		error_x.copyTo(person.error_x);
		error_y.copyTo(person.error_y);
	}
	else{
		person.error_x += error_x;
		person.error_y += error_y;
	}

}

void send_detected_message(lcm_t* lcm, person& person){
	// pack a person into an object_detected message,
	// the vectors x,y,z are used as error vectors
	mavlink_message_t ptracking;
	mavlink_object_detected_t pd;
	pd.timestamp = current_time;
	//pd.identifier = "person";
	pd.position[0] = person.position.at<float>(0);
	pd.position[1] = person.position.at<float>(1);
	pd.position[2] = person.position.at<float>(2);
	pd.vector_x[0] = person.error_x.at<float>(0);
	pd.vector_x[1] = person.error_x.at<float>(1);
	pd.vector_x[2] = person.error_x.at<float>(2);
	pd.vector_y[0] = person.error_y.at<float>(0);
	pd.vector_y[1] = person.error_y.at<float>(1);
	pd.vector_y[2] = person.error_y.at<float>(2);
	pd.vector_z[0] = person.height;
	pd.vector_z[1] = person.error_z.at<float>(1);
	pd.vector_z[2] = person.error_z.at<float>(2);
	pd.tracking_id = person.id;
	mavlink_msg_object_detected_encode(getSystemID(), compid, &ptracking,&pd);
	sendMAVLinkMessage(lcm, &ptracking);
	//printf("msg for person: %i\n", person.id);
}

void patch_detection(cv::Mat& image, std::vector<person>& persons){
	// this method uses the image-patches stored with each person to relocate a person within a frame
	// it updates the position and image-patch of said person
	cv::Mat overlay = cv::Mat::zeros(480,640,CV_8UC1);
	int u,v;
	std::vector<person>::iterator p = persons.begin();
	while(p != persons.end())
	{
		if(p->last_timestamp != current_time){
			// if the current call originates from the HOG-detector,
			// skip all person directly originating from the HOG-detector
			try{
				//position in image
				world_to_pixel(p->position,u,v);
				// check if person in the image at all
				if(u < 50 || u > img_width - 50 || v > img_height || v < (p->patch).size().height){
					// person is not in (completely) in the frame -> delete
					// not that useful for static camera
					p = persons.erase(p);
					continue;
				}
				// points for image-roi
				cv::Point2i img_tl(std::max(cvRound(u - (p->patch).size().width *0.7),0),
						std::max(cvRound(v- (p->patch).size().height *1.05),0));

				cv::Point2i img_br(std::min(cvRound(u + (p->patch).size().width * 0.7),img_width),
						std::min(cvRound(v - (p->patch).size().height *0.55),img_height));

				cv::Point2i pers_tl(0,0);
				cv::Point2i pers_br(((p->patch).cols-1),cvRound((p->patch).rows*0.4));

				// grey box indiacting the area to search for patch
				cv::rectangle(overlay, img_tl, img_br, cv::Scalar(127), 3);

				cv::Mat roi_to_search = image(cv::Rect(img_tl,img_br));
				cv::Mat roi_person = (p->patch)(cv::Rect(pers_tl, pers_br));
				cv::Mat result;

				result.create(roi_to_search.cols - roi_person.cols + 1, roi_to_search.rows - roi_person.rows + 1,CV_32FC1);
				/* Method:
				 * 0: SQDIFF
				 * 1: SQDIFF NORMED
				 * 2: TM CCORR
				 * 3: TM CCORR NORMED
				 * 4: TM COEFF
				 * 5: TM COEFF NORMED
				 * */
				cv::matchTemplate(roi_to_search, roi_person, result, 0);
				// evaluate the results
				double minVal; double maxVal; cv::Point minLoc; cv::Point maxLoc;
				cv::Point matchLoc;
				cv::minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat() );
				matchLoc = minLoc;
				// update person
				// update pixel location of top center
				img_tl += matchLoc;
				img_tl += cv::Point2i(cvRound((p->patch).cols/2),0);
				// compute position of top of the head
				//printf("was there: x:%f, y:%f, z:%f\n",(p->position).at<float>(0),(p->position).at<float>(1),(p->position).at<float>(2));
				int a,b;
				world_to_pixel(p->position,a,b);
				cv::circle(overlay,cv::Point2i(a,b),5, cv::Scalar(100),2);
				//cv::circle(image,cv::Point2i(a,b-30),5, cv::Scalar(50),2);
				pixel_to_world( img_tl.x,img_tl.y + cvRound((p->patch).rows * 0.2) ,p->height*0.8, p->position);
				//printf("size: %f\n",p->height);
				// set position to the ground
				//printf("the guy's at x:%f, y:%f, z:%f\n",(p->position).at<float>(0),(p->position).at<float>(1),(p->position).at<float>(2));
				(p->position).at<float>(2) = 0;

				// calculate new patch height
				// adjust patch width by new_patch_height / old_patch_height
				//printf("person was here: u:%d, v:%d\n",u,v);
				//printf("person shifted by this: d_u:%d, d_v:%d\n",minLoc.x,minLoc.y);

				//cv::circle(image,cv::Point2i(u,v),5, cv::Scalar(50),2);
				// not so happy but +- works
				//cv::circle(image,img_tl,10, cv::Scalar(0),3);

				int s,t;
				world_to_pixel(p->position,s,t);
				float scaling_factor = (float)(t - (img_tl.y + cvRound((p->patch).rows * 0.2)) ) / ((p->patch).rows*0.8);
				cv::Point2i new_tl( std::max( cvRound( img_tl.x - ((p->patch).cols * scaling_factor) / 2), 0),
									std::max( img_tl.y, 0));

				cv::Point2i new_br( std::min( cvRound( s + ((p->patch).cols * scaling_factor) / 2), img_width),
									std::min( t, img_height));
				//printf("scaling: %f\n", scaling_factor);
				p->confidence = (float)(1 - (minVal / maxVal));
				printf("confidence: %f\n",p->confidence);
				//printf("height: %f\n",p->height);
				// update patch
				cv::rectangle(overlay, new_tl, new_br, cv::Scalar(200), 3);
				cv::Mat updated_patch = image(cv::Rect(new_tl, new_br));
				updated_patch.copyTo(p->patch);
				//update errors
				set_detection_errors(*p,3,3,0);
				p->last_timestamp = current_time;
				if((p->patch).cols < 75 || (p->patch).rows < 75 || p->confidence < confidence_threshold){
					// set timer to 0.3 secs in future or next scheduled hog detection
					hog_timer = std::min((current_time - last_hog_detection) + 300000,hog_timer);
					p = persons.erase(p);
					continue;
				}
			}
			catch (...){
				// some error
				p = persons.erase(p);
				continue;
			}

		}
		// else the detection comes right from the hog detector

		// other stuff
		p++;
	}

	// remove duplicates
	p = persons.begin();
	bool erase = false;
	while(p != persons.end()){
		erase = false;
		world_to_pixel(p->position,u,v);
		cv::Rect person_to_test = cv::Rect( cv::Point2i(cvRound(u - ( (p->patch).cols)/2) , v - (p->patch).rows),
											cv::Point2i(cvRound(u + ( (p->patch).cols)/2), v));
		//cv::rectangle(image,person_to_test,cv::Scalar(255), 3);
		for( std::vector<person>::iterator j = persons.begin(); j != persons.end(); j++ ){
			int m,n;
			if (j != p){
				world_to_pixel(j->position,m,n);
				cv::Rect person_to_compare = cv::Rect( cv::Point2i(cvRound(m - ( (j->patch).cols)/2) , n - (j->patch).rows),
													   cv::Point2i(cvRound(m + ( (j->patch).cols)/2), n));
				if( ( (float)person_to_test.area() / ( person_to_test | person_to_compare ).area() ) > 0.8){
					erase = true;
					//cv::rectangle(image, (person_to_test | person_to_compare),cv::Scalar(255), 3);
					//printf("overlap: %f\n",(float)person_to_test.area() / ( person_to_test | person_to_compare ).area());
				}
			}
		}
		if(erase){
			p = persons.erase(p);
			continue;
		}
		else{
			p++;
		}
	}
	cv::addWeighted(image,1,overlay, -1,0,image);
	try{
		if(!persons.empty() && display){
			cv::namedWindow("person");
			cv::imshow("person",persons[0].patch);
		}
	}
	catch (...){;}

}

void hog_detection(cv::Mat& image, std::vector<person>& persons){
	// this method uses the standard hog-detector from opencv with the default descriptor.
	// it runs the hog detector on a scaled down image (320x240 at the moment) and then verifies
	// each detected person using the distance (and soon) the estimated height.

	//printf("hog detection\n");
	int u,v;
	last_hog_detection = current_time;
	last_hog_yaw = current_yaw;
	cv::Mat resized(cvRound(img_width / 2),cvRound(img_height / 2), CV_8UC1);
	cv::resize(image, resized,cv::Size(), 1./2, 1./2);
	std::vector<cv::Rect> found, found_filtered;
	// run the detector with default parameters. to get a higher hit-rate
	// (and more false alarms, respectively), decrease the hitThreshold and
	// groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
	hog.detectMultiScale(resized, found, 0, cv::Size(8,8), cv::Size(32,32), 1.05, 2);

	size_t i, j, k;
	for( i = 0; i < found.size(); i++ )
	{
		cv::Rect r = found[i];
		for( j = 0; j < found.size(); j++ )
		{
			if( j != i && ( (float)r.area() / (r | found[j]).area() > 0.8) ){
				//cv::rectangle(image, r | found[j] ,cv::Scalar(0), 3);
				break;
			}
		}
		if( j == found.size() )
			found_filtered.push_back(r);
	}
	for( i = 0; i < found_filtered.size(); i++ )
	{
		bool found_match=false;
		cv::Rect r = found_filtered[i];
		// the HOG detector returns slightly larger rectangles than the real objects.
		// so we slightly shrink the rectangles to get a nicer output.
		r.x += cvRound(r.width*0.2);
		r.width = cvRound(r.width*0.6);
		r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height*0.85);

		// double coordinates and size for alignment in original image size
		r.x *=2;
		r.y *=2;
		r.width *=2;
		r.height *=2;

		// clip rectangles to image
		r.x = std::max(r.x , 0);
		r.y = std::max(r.y , 0);
		r.width = std::min(r.width, img_width - r.x );
		r.height = std::min(r.height, img_height - r.y);

		//cv::rectangle(image, r.tl(), r.br(), cv::Scalar(0), 3);

		// compute position of detected person
		u = r.x + r.width / 2;
		v = r.y + r.height;
		cv::Mat detected = (cv::Mat_<float>(4,1) <<
				u,
				v,
				1,
				1.
				);

		pixel_to_world(cvRound(detected.at<float>(0)),cvRound(detected.at<float>(1)),0.,detected);
		// validate the detection and compare position with the entries in persons.
		// if there's a match update the person
		// otherwise add a new person
		person new_guy;
		detected.copyTo(new_guy.position);
		new_guy.height = estimate_height(detected, cvRound( ( r.tl().x + r.br().x ) /2 ), r.tl().y);
		if(!validate_detected_person(new_guy)){
			continue;
		}
		for (k = 0; k < persons.size(); k++)
		{
			// check for matches
			if (sqrt(pow((persons[k].position.at<float>(0) - detected.at<float>(0,0)),2) + pow((persons[k].position.at<float>(1) - detected.at<float>(1,0)),2)) <=
					max_speed * (current_time - persons[k].last_timestamp) / 1000000 ){

				cv::Mat roi = image(cv::Rect(r.tl(),r.br()));
				roi.copyTo(persons[k].patch);
				persons[k].last_timestamp = current_time;
				detected.copyTo(persons[k].position);
				persons[k].height = estimate_height(detected, cvRound((r.tl().x + r.br().x)/2), r.tl().y );
				set_detection_errors(persons[k],detection_error_y,detection_error_x,1);
				persons[k].confidence = 1.;
				found_match = true;
				break;
			}
		}
		if(!found_match){
			// no match -> add to persons
			cv::Mat roi = image(cv::Rect(cv::Point(r.tl().x,r.tl().y),cv::Point(r.br().x,r.br().y)));
			roi.copyTo(new_guy.patch);
			new_guy.last_timestamp = current_time;
			new_guy.confidence = 1.;
			new_guy.id = ident_number++;

			//printf("estimated height: %f\n",new_guy.height);
			set_detection_errors(new_guy,detection_error_y,detection_error_x,1);
			persons.push_back(new_guy);
		}
	}
	// cost neglectable, improves tracking for partially occluded people
	patch_detection(image, persons);
}

/**
 * @brief Handle incoming MAVLink packets containing images
 *
 */
void image_worker()
{
	//const mavlink_message_t* msg = getMAVLinkMsgPtr(container);
	// Pointer to shared memory data
	//PxSHMImageClient* client = static_cast<PxSHMImageClient*>(user);
	//lcm_t* lcm = static_cast<lcm_t*>(user);

	//printf("GOT IMG MSG\n");
	cv::Mat img;
	cv::Mat imgToSave;

	// read mono image data
        if (ImageClient.readMonoImage(&imageMessage, img))
	{
		//////////////////////////////////////

		// APPLY ONE OF THE OPENCV FUNCTIONS HERE, AND OUTPUT IMAGE HERE

		//////////////////////////////////////

        newImage = false;
		// setup work for coordinate transforms etc
		img_width = img.cols;
		img_height = img.rows;
		float px,py,pz;
		float ro,pi,ya;
        current_time = PxSHMImageClient::getTimestamp(&imageMessage);
        PxSHMImageClient::getGroundTruth(&imageMessage, px, py, pz);
        PxSHMImageClient::getRollPitchYaw(&imageMessage, ro, pi, ya);
		// for webcam testing
		pz = cam_height;
		px = 0;
		py = 0;

		image_mutex.unlock();

		// assemble matrices for coord transformations
		set_rot_trans_matrices(px, py, pz, ro, pi, ya);

		if ((current_time - last_hog_detection) > hog_timer
			|| abs(current_yaw - last_hog_yaw) > threshold_yaw
			|| (persons.size() == 0 && (current_time - last_hog_detection) > 500000)){
			// test if its time for a new hog detection
			request_redetection = true;
		}

		if(request_redetection){
			double t = (double)cv::getTickCount();
			hog_detection(img, persons);
			t = (double)cv::getTickCount() - t;
			double msec = t*1000./cv::getTickFrequency();

			if(verbose){
				printf("detection time = %gms for hog\n", msec);
			}
			// reset timer
			request_redetection = false;
			hog_timer = 1000000;

			if(logging){
				// logging into file
				char output[100];
				// format is trajectory id, timestamp, raw_x, raw_y, predicted_x, predicted_y, corrected_x, corrected_y,
				// though as this is the initial recording position, prediction and corrected are the same
				std::sprintf(output,"%llu,%f,,\n",current_time,msec);

				log_file << std::string(output);
			}
		}
		// patch based detection
		else {
			double t = (double)cv::getTickCount();
			int p_size = persons.size();
			patch_detection(img,persons);
			t = (double)cv::getTickCount() - t;
			double msec = t*1000./cv::getTickFrequency();

			if(verbose){
				printf("detection time = %gms for patch matching\n", msec);
			}

			if(logging){
						// logging into file
						char output[100];
						// format is trajectory id, timestamp, raw_x, raw_y, predicted_x, predicted_y, corrected_x, corrected_y,
						// though as this is the initial recording position, prediction and corrected are the same
						std::sprintf(output,"%llu,,%f,%d\n",current_time,msec,p_size);

						log_file << std::string(output);
					}
		}
		// send out the results
		for (std::vector<person>::iterator p = persons.begin();p != persons.end(); p++){
			send_detected_message(lcm, *p);

		}
		// test
//		int q,w;
//		cv::Mat pos1 = (cv::Mat_<float>(4,1)<< 5,1.5,0,1);
//		world_to_pixel(pos1,q,w);
//		cv::circle(img,cv::Point2i(q,w),5, cv::Scalar(50),2);
//		cv::Mat pos2 = (cv::Mat_<float>(4,1)<< 3,0,0,1);
//		world_to_pixel(pos2,q,w);
//		cv::circle(img,cv::Point2i(q,w),5, cv::Scalar(50),2);
//		cv::Mat pos3 = (cv::Mat_<float>(4,1)<< 5,0,0,1);
//		world_to_pixel(pos3,q,w);
//		cv::circle(img,cv::Point2i(q,w),5, cv::Scalar(50),2);

		struct timeval tv;
		gettimeofday(&tv, NULL);
		uint64_t currTime = ((uint64_t)tv.tv_sec) * 1000000 + tv.tv_usec;
                uint64_t timestamp = PxSHMImageClient::getTimestamp(&imageMessage);

		uint64_t diff = currTime - timestamp;

		if (verbose)
		{
                        fprintf(stderr, "# INFO: Time from capture to display: %llu ms for camera %llu\n", diff / 1000, PxSHMImageClient::getCameraID(&imageMessage));
		}

		// Display if switched on
		if(display){
#ifndef NO_DISPLAY
			if ((ImageClient.getCameraConfig() & PxSHM::CAMERA_FORWARD_LEFT) == PxSHM::CAMERA_FORWARD_LEFT)
			{
				cv::namedWindow("Left Image (Forward Camera)");
				cv::imshow("Left Image (Forward Camera)", img);
			}
			else
			{
				cv::namedWindow("Left Image (Downward Camera)");
				cv::imshow("Left Image (Downward Camera)", img);
			}
#endif
		}

		img.copyTo(imgToSave);


#ifndef NO_DISPLAY
		int c = cv::waitKey(3);
		switch (static_cast<char>(c))
		{
		case 'f':
		{
			char index[20];
			sprintf(index, "%04d", imageCounter++);
			cv::imwrite(std::string(fileBaseName+index+fileExt).c_str(), imgToSave);
		}
		break;
		default:
			break;
		}
#endif
	}
        else
        {
        	newImage = false;
        	image_mutex.unlock();
        }
}




// TODO remove this whole (mavlink handler) stuff
static void
mavlink_handler (const lcm_recv_buf_t *rbuf, const char * channel,
		const mavconn_mavlink_msg_container_t* container, void * user)
{
	const mavlink_message_t* msg = getMAVLinkMsgPtr(container);
	mavlink_message_t response;
	lcm_t* lcm = static_cast<lcm_t*>(user);
	//printf("Received message #%d on channel \"%s\" (sys:%d|comp:%d):\n", msg->msgid, channel, msg->sysid, msg->compid);
// paramclient call


	switch(msg->msgid)
	{
	uint32_t receiveTime;
	uint32_t sendTime;
	/*case MAVLINK_MSG_ID_COMMAND_SHORT:
	{
		mavlink_command_short_t cmd;
		mavlink_msg_command_short_decode(msg, &cmd);
		printf("Message ID: %d\n", msg->msgid);
		printf("Command ID: %d\n", cmd.command);
		printf("Target System ID: %d\n", cmd.target_system);
		printf("Target Component ID: %d\n", cmd.target_component);
		printf("\n");

		if (cmd.confirmation)
		{
			printf("Confirmation requested, sending confirmation:\n");
			mavlink_command_ack_t ack;
			ack.command = cmd.command;
			ack.result = 3;
			mavlink_msg_command_ack_encode(getSystemID(), compid, &response, &ack);
			sendMAVLinkMessage(lcm, &response);
		}
	}
	break;*/
	case MAVLINK_MSG_ID_ATTITUDE:
		gettimeofday(&tv, NULL);
		receiveTime = tv.tv_usec;
		sendTime = mavlink_msg_attitude_get_time_boot_ms(msg);
		//printf("Received attitude message, transport took %f ms\n", (receiveTime - sendTime)/1000.0f);
		break;
	case MAVLINK_MSG_ID_GPS_RAW_INT:
	{
		mavlink_gps_raw_int_t gps;
		mavlink_msg_gps_raw_int_decode(msg, &gps);
		//printf("GPS: lat: %f, lon: %f, alt: %f\n", gps.lat/(double)1E7, gps.lon/(double)1E7, gps.alt/(double)1E6);
		break;
	}
	case MAVLINK_MSG_ID_RAW_PRESSURE:
	{
		mavlink_raw_pressure_t p;
		mavlink_msg_raw_pressure_decode(msg, &p);
		//printf("PRES: %f\n", p.press_abs/(double)1000);
	}
	break;
	default:
		//printf("ERROR: could not decode message with ID: %d\n", msg->msgid);
		break;
	}
}

static void image_handler(const lcm_recv_buf_t* rbuf, const char* channel,
						  const mavconn_mavlink_msg_container_t* container,
						  void* user)
{
	const mavlink_message_t* msg = getMAVLinkMsgPtr(container);
	image_mutex.lock();
	memcpy(&imageMessage, msg, sizeof(mavlink_message_t));
	newImage = true;
	image_cond->signal();
	image_mutex.unlock();
}

void* lcm_wait(void* lcm_ptr)
												{
	lcm_t* lcm = (lcm_t*) lcm_ptr;
	// Blocking wait for new data
	while (1)
	{
		lcm_handle (lcm);
	}
	return NULL;
												}

// Handling Program options
static GOptionEntry entries[] =
{
		{ "sysid", 'a', 0, G_OPTION_ARG_INT, &sysid, "ID of this system, 1-255", "42"},
		{ "compid", 'c', 0, G_OPTION_ARG_INT, &compid, "ID of this component, 1-255", "112" },
		{ "verbose", 'v', 0, G_OPTION_ARG_NONE, &verbose, "Be verbose", (verbose) ? "true" : "false" },
		{ "logging", 'l', 0, G_OPTION_ARG_NONE, &logging, "Log to file", (logging) ? "true" : "false" },
		{ "display", 'd', 0, G_OPTION_ARG_NONE, &display, "Display video stream and raw detections", (display) ? "true" : "false" },
		{ "config", 'g', 0, G_OPTION_ARG_STRING, config_file, "Filename of paramClient config file", "../conf/main.cfg"},
		{ NULL }
};

int main(int argc, char* argv[])
{
	GError *error = NULL;
	GOptionContext *context;

	context = g_option_context_new ("- localize based on natural features");
	g_option_context_add_main_entries (context, entries, "Localization");
	if (!g_option_context_parse (context, &argc, &argv, &error))
	{
		g_print ("Option parsing failed: %s\n", error->message);
		exit(EXIT_FAILURE);
	}
	g_option_context_free(context);
	// Handling Program options


	lcm = lcm_create("udpm://");
	if (!lcm)
	{
		fprintf(stderr, "# ERROR: Cannot initialize LCM.\n");
		exit(EXIT_FAILURE);
	}

	mavconn_mavlink_msg_container_t_subscription_t * comm_sub =
			mavconn_mavlink_msg_container_t_subscribe (lcm, MAVLINK_MAIN, &mavlink_handler, lcm);

	// Thread
	GThread* lcm_thread;
	GError* err;

	if( !g_thread_supported() )
	{
		g_thread_init(NULL);
		// Only initialize g thread if not already done
	}
	if (!image_cond)
	{
		image_cond = new Glib::Cond;
	}

	if( (lcm_thread = g_thread_create((GThreadFunc)lcm_wait, (void *)lcm, TRUE, &err)) == NULL)
	{
		printf("Thread create failed: %s!!\n", err->message );
		g_error_free ( err ) ;
	}

	ImageClient.init(true, PxSHM::CAMERA_FORWARD_LEFT);
	paramClient = new MAVConnParamClient(getSystemID(), compid, lcm, config_file->str, verbose);
	// init hog-detector
	hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

	// Ready to roll
	fprintf(stderr, "# INFO: Image client ready, waiting for images..\n");

	// Subscribe to MAVLink messages on the image channel
    mavconn_mavlink_msg_container_t_subscription_t* imgSub = mavconn_mavlink_msg_container_t_subscribe(lcm, MAVLINK_IMAGES, &image_handler, &ImageClient);

	//imageServer.init(sysid, compid, lcm, PxSHM::CAMERA_FORWARD_LEFT);

	signal(SIGINT, signalHandler);

	// prepare for logging
		if(logging){
			time_t now = time(NULL);
			char log_name[50];
			strftime(log_name,50,"../log/log_detector_%d%m%Y_%H:%M.csv",localtime(&now));
			log_file.open(std::string(log_name));
			if(!log_file.is_open()){
				printf("log file could not be created, no logging takes place\n");
				logging = false;
			}
		}

	while (!quit)
		{
			update_parameters();
			image_mutex.lock();
			while (!newImage){
				image_cond->wait(image_mutex);
			}
			image_mutex.unlock();
			image_worker();
		}

	log_file.close();
	mavconn_mavlink_msg_container_t_unsubscribe(lcm, imgSub);
	mavconn_mavlink_msg_container_t_unsubscribe (lcm, comm_sub);
	lcm_destroy (lcm);
	g_thread_join(lcm_thread);
	return 0;
}

