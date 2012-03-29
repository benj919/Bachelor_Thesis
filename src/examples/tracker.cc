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
 *   @author Lorenz Meier <mavteam@student.ethz.ch>
 *
 */

#include <cstdio>
#include <unistd.h>
#include <glib-2.0/glib.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <mavconn.h>
#include <mavlink.h>
#include <core/MAVConnParamClient.h>
#include <object_detection/object_detection.h>
#include <interface/shared_mem/PxSHMImageClient.h>

// Latency Benchmarking
#include <sys/time.h>
#include <time.h>

// Timer for benchmarking
struct timeval tv;

// Setup for lcm decoupling
PxSHMImageClient ImageClient;
MAVConnParamClient* paramClient;

// flags
bool quit =false;
bool debug = false;
bool verbose = false;
bool logging = false;
bool overlay = false;
bool video = false;

// logs etc
std::ofstream log_file;
static GString* config_file = g_string_new("../conf/main.cfg");

/*
 * The structure trajectory contains:
 * an id to identify it
 * a vector of points, containing the positions
 * two vectors error_a and error_b containing
 * the axes for the uncertainty ellipse
 */
struct trajectory{
	uint64 last_update;
	int last_tracking_id;
	int id;
	bool active;
	float height;
	std::vector<cv::Mat> points;
	std::vector<cv::Mat> raw_points;
	cv::KalmanFilter kalman;
	cv::Mat prediction;
	cv::Scalar color;
};

std::vector<trajectory> people_trajectories;

// video capture
cv::VideoWriter video_writer;

// setup
int sysid = 42;
int compid = 111;
int color_cycle = 0;
float max_speed;
int trajectory_number = 0;


// coordinate transform
cv::Mat W2B,B2C,C2P;
cv::Mat pos_pixhawk;

// image info/data
int img_width,img_height;
int mx,my;
float camera_height;
float f;
float principle_x, principle_y;
uint64 last_image_frame, last_detection_frame;
cv::Mat background, current_image;

//rectangular confinement
float x_space = 10.;
float y_space = 10.;
int max_pixel = 500;
int u_pixel = 420;
int v_pixel = 560;

// statistics
float min_delay = 1000;
float max_delay = 0;
float avg_delay = 0;
float last_delay = 0;
double sum_delay = 0;
int count_delay = 1;


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

cv::Scalar next_color(){
	// returns the next color out of a list of six colors
	// format: Opencv Scalar bgr
	color_cycle = (color_cycle + 1) % 6;
	switch (color_cycle){
	case 0:
		//red
		return cv::Scalar(0,0,255);
	case 1:
		//green
		return cv::Scalar(0,255,0);
	case 2:
		//blue
		return cv::Scalar(255,0,0);
	case 3:
		//yellow
		return cv::Scalar(0,255,255);
	case 4:
		//magenta
		return cv::Scalar(255,0,255);
	case 5:
		//cyan
		return cv::Scalar(255,255,0);
	default:
		//white
		return cv::Scalar(255,255,255);
	}


}


void print_mat(cv::Mat& mat ){
	// for printing 1channel float mats
    int i, j;
    for( i = 0; i < mat.rows; i++ )
    {
        for( j = 0; j < mat.cols; j++ )
        {
            printf( "%f ", mat.at<float>( i, j ) );
        }
        printf( "\n" );
    }
}

cv::Point2i point_max(cv::Point2i a, cv::Point2i b){
	// returns the point on the bottom right
	// of a rectangle spanned by 2 points a & b
	int c = std::max(a.x,b.x);
	int d = std::max(a.y,b.y);
	cv::Point2i r(c,d);
	return r;
}

cv::Point2i point_min(cv::Point2i a, cv::Point2i b){
	// returns the point on the top left
	// of a rectangle spanned by 2 points a & b
	int c = std::min(a.x,b.x);
	int d = std::min(a.y,b.y);
	cv::Point2i r(c,d);
	return r;
}

void update_parameters(){
	max_speed = paramClient->getParamValue("MAX_SPEED");
	x_space = cvRound(paramClient->getParamValue("X_SPACE"));
	y_space = cvRound(paramClient->getParamValue("Y_SPACE"));
	pos_pixhawk = (cv::Mat_<float>(4,1) << 0,0,paramClient->getParamValue("CAM_HEIGHT"),0);
	mx = cvRound(paramClient->getParamValue("CAMERA_MX"));
	my = cvRound(paramClient->getParamValue("CAMERA_MY"));
	f = paramClient->getParamValue("CAMERA_F");
	camera_height = paramClient->getParamValue("CAM_HEIGHT");
	v_pixel = cvRound(u_pixel * 2 * y_space / x_space);
	principle_x = paramClient->getParamValue("PRINCIPLE_X");
	principle_y = paramClient->getParamValue("PRINCIPLE_Y");
}

float p2p_distance(cv::Mat& pos_a, cv::Mat& pos_b){
	// returns the distance between two 3d-points
	float root =(pos_a.at<float>(0)-pos_b.at<float>(0))*(pos_a.at<float>(0)-pos_b.at<float>(0))
			+(pos_a.at<float>(1)-pos_b.at<float>(1))*(pos_a.at<float>(1)-pos_b.at<float>(1))
			+(pos_a.at<float>(2)-pos_b.at<float>(2))*(pos_a.at<float>(2)-pos_b.at<float>(2));
	return sqrt(root);
}

cv::Point2i coord_to_point(cv::Mat& position){
	// use for 2D map
	// converts a point in world coordinates to a pixel on a 2d-map
	if(overlay){
		return cv::Point2i( cvRound( v_pixel / 2 - ( position.at<float>(1) / y_space * v_pixel/2)), cvRound( (position.at<float>(0) / x_space * u_pixel) ) );
	}
	else{
		return cv::Point2i( cvRound( 240 / 2 - (position.at<float>(1) / y_space * 240/2)), cvRound( (position.at<float>(0) / x_space * 360) ) );
	}
}

cv::Point2i world_to_pixel(cv::Mat& position){
	// use for projection onto camera-image
	// Takes a point in 3D space and returns its pixel-coordinates
	// position of the bottom-center pixel
	cv::Mat bc = /* C2P* */(B2C * (W2B * ( position - pos_pixhawk )));
	bc = (f / (bc.at<float>(2)) * bc);
	bc.at<float>(3) = 1;
	bc = C2P * bc;
	return cv::Point2i(cvRound(bc.at<float>(0)), cvRound(bc.at<float>(1)));
}

void correlate_to_trajectory(int id, uint64 detection_time, cv::Mat& d4_position, cv::Mat& error_a, cv::Mat& error_b, float height){
	cv::Mat position = ( cv::Mat_<float>(3,1) << d4_position.at<float>(0),d4_position.at<float>(1),d4_position.at<float>(2));
	trajectory* best_match;
	bool found_one = false;
	float best_distance = 1000;

	// update all predictions and transition matrices if measurement for new frame
	if(last_detection_frame != detection_time){
		for(std::vector<trajectory>::iterator trajectory = people_trajectories.begin();trajectory != people_trajectories.end(); ++trajectory){
			if(!(trajectory->active) ){
				continue;
			}
			float dt = ( detection_time - trajectory->last_update ) / 1000000; // dt in seconds, i hope
			cv::Mat new_transition_matrix = (cv::Mat_<float>(6,6) <<
					1,0,0,dt,0,0,
					0,1,0,0,dt,0,
					0,0,1,0,0,dt,
					0,0,0,1,0,0,
					0,0,0,0,1,0,
					0,0,0,0,0,1
			);
			new_transition_matrix.copyTo( (trajectory->kalman).transitionMatrix);
			//cv::Mat prediction = cv::KalmanPredict(trajectory->kalman,0);
			cv::Mat prediction = (trajectory->kalman).predict();
			prediction.copyTo(trajectory->prediction);
		}
		last_detection_frame = detection_time;
	}
	// check if there are any trajectories already
	for(std::vector<trajectory>::iterator trajectory = people_trajectories.begin();trajectory != people_trajectories.end(); ++trajectory){

		if(trajectory->last_update == detection_time || !(trajectory->active) ){
			// already assigned a new point to the trajectory
			continue;
		}
		else if(trajectory->last_update + 3000000 <= detection_time){
			trajectory->active = false;
			continue;
		}

		float dist =  p2p_distance((trajectory->raw_points).back(),position);
		float allowed = ((float)(detection_time - trajectory->last_update)) / 1000000 * max_speed;
		allowed += fabs(error_a.at<float>(0)) + fabs(error_b.at<float>(1));
		//printf("error: %f %f dist: %f ",fabs(error_a.at<float>(0)),fabs(error_b.at<float>(1)), dist);
		//printf("allowed: %f\n", allowed);
		if(trajectory->last_tracking_id == id){
			// trust the tracking id for now
			best_match = &(*trajectory);
			found_one = true;
			break;
		}
		else if( (best_distance > dist) && dist < allowed){
			// update best_match

			best_match = &(*trajectory);
			best_distance = dist;
			found_one = true;
		}
		else{}
	}
	if(found_one){

		cv::Mat error_matrix = (cv::Mat_<float>(3,3) <<
									std::max(pow(error_a.at<float>(0),2),pow(error_b.at<float>(0),2)),0,0,
									0,std::max(pow(error_a.at<float>(1),2),pow(error_b.at<float>(1),2)),0,
									0,0,0);
		error_matrix.copyTo((best_match->kalman).measurementNoiseCov);

		// update trajectory
		cv::Mat filtered_position;
		(best_match->kalman).correct(position).copyTo(filtered_position);
		cv::Mat d4_filtered_position = ( cv::Mat_<float>(4,1) << filtered_position.at<float>(0),filtered_position.at<float>(1),filtered_position.at<float>(2),1);
		(best_match->points).push_back(d4_filtered_position);
		(best_match->raw_points).push_back(d4_position);
		best_match->last_tracking_id = id;
		best_match->last_update = detection_time;
		best_match->height = height;
		if(logging){
			// logging into file
			char output[300];
			// format is trajectory id, timestamp, raw_x, raw_y, predicted_x, predicted_y, corrected_x, corrected_y, error_x, error_y, delay
			std::sprintf(output,"%d,%llu,%4.2f,%4.2f,%4.2f,%4.2f,%4.2f,%4.2f,%4.2f,%4.2f,%4.2f\n",
						best_match->id,best_match->last_update,
						position.at<float>(0),position.at<float>(1),
						(best_match->prediction).at<float>(0),(best_match->prediction).at<float>(1),
						filtered_position.at<float>(0),filtered_position.at<float>(1),
						error_a.at<float>(0),error_b.at<float>(1),
						last_delay);
			log_file << std::string(output);
		}
	}
	else{
		// create new trajectory
		trajectory new_traj;
		new_traj.points.push_back(d4_position);
		new_traj.raw_points.push_back(d4_position);
		new_traj.id = trajectory_number++;
		new_traj.last_tracking_id = id;
		new_traj.last_update = detection_time;
		new_traj.active = true;
		new_traj.color = next_color();
		new_traj.height = height;
		new_traj.kalman.init(6,3,0,CV_32F);
		cv::setIdentity( new_traj.kalman.measurementMatrix );

		cv::Mat error_matrix = (cv::Mat_<float>(3,3) <<
									std::max(pow(error_a.at<float>(0),2),pow(error_b.at<float>(0),2)),0,0,
									0,std::max(pow(error_a.at<float>(1),2),pow(error_b.at<float>(1),2)),0,
									0,0,0);
		error_matrix.copyTo(new_traj.kalman.measurementNoiseCov);

		cv::setIdentity( (new_traj.kalman).processNoiseCov, cvRealScalar(1e-3) ); // ignore or experiment
		cv::setIdentity((new_traj.kalman).errorCovPost, cv::Scalar::all(1));


		cv::Mat initial_state = (cv::Mat_<float>(6,1) << position.at<float>(0), position.at<float>(1), 0,0,0,0);
		initial_state.copyTo( new_traj.kalman.statePost);

		people_trajectories.push_back(new_traj);

		if(logging){
			// logging into file
			char output[300];
			// format is trajectory id, timestamp, raw_x, raw_y, predicted_x, predicted_y, corrected_x, corrected_y,
			// though as this is the initial recording position, prediction and corrected are the same
			std::sprintf(output,"%d,%llu,%4.2f,%4.2f,%4.2f,%4.2f,%4.2f,%4.2f,%4.2f,%4.2f,%4.2f\n",
					new_traj.id,new_traj.last_update,
					position.at<float>(0),position.at<float>(1),
					position.at<float>(0),position.at<float>(1),
					position.at<float>(0),position.at<float>(1),
					error_a.at<float>(0),error_b.at<float>(1),
					last_delay);

			log_file << std::string(output);
		}
	}
}

void plot_trajectories(){
	// plot 1m x 1m scale to top left
	if(overlay){
		current_image = cv::Mat::zeros(480,640,CV_8UC3);
		cv::Mat overlay_roi = current_image(cv::Rect(320 - v_pixel / 2 ,30,v_pixel,u_pixel));
		cv::cvtColor(background, current_image, CV_GRAY2RGB);
		cv::Mat overlay_image(u_pixel, v_pixel, CV_8UC3, cv::Scalar(0,0,0));
		cv::Mat lower_ribbon = cv::Mat::zeros(30,640,CV_8UC3);
		cv::Mat ribbon_roi = current_image(cv::Rect(0,450,640,30));
		int counter=0;
		std::vector<trajectory>::iterator trajectory = people_trajectories.begin();
		while(trajectory != people_trajectories.end()){
			// plot starting point
			if(!(trajectory->active)){
				trajectory++;
				continue;
			}
			if( trajectory->last_update + 3000000 < last_image_frame){
				trajectory->active = false;
				trajectory++;
				continue;
			}
			counter +=1;
			cv::Point2i last_point, current_point;
			cv::Point2i last_raw_point, current_raw_point;
			last_point = coord_to_point((trajectory->points)[0]);
			last_raw_point = coord_to_point((trajectory->raw_points)[0]);
			//printf("point: %d %d\n",last_point.x,last_point.y);
			//printf("coords: %f %f\n",(trajectory->points)[0].at<float>(0),(trajectory->points)[0].at<float>(1));
			//printf("x_space: %f, y_space: %f, x_pixel: %d, y_pixel: %d \n",x_space, y_space, x_pixel, y_pixel);
			//cv::circle(image,last_point,5, cv::Scalar(255),2);

			// plot path
			int size = trajectory->points.size();
			if(size > 1){
				for(int i = 1; i < size; i++){
					current_point = coord_to_point((trajectory->points)[i]);
					current_raw_point = coord_to_point((trajectory->raw_points)[i]);
					//cv::circle(image,current_point,10, cv::Scalar(255),1);
					cv::line(overlay_image,last_point,current_point,trajectory->color,1);
					//cv::line(image,last_raw_point,current_raw_point,trajectory->color,1);
					last_point = current_point;
					last_raw_point = current_raw_point;
				}
				cv::circle(overlay_image,current_point,5, trajectory->color,2);
				//cv::circle(image,current_raw_point,5, trajectory->color,2);
				//cv::putText(image, "trajectory "  , current_point + cv::Point2i(10,10), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255,255,255), 1, 8, false);
				//printf("world: %f %f \n",(trajectory->points).back().at<float>(0),(trajectory->points).back().at<float>(1));
				//printf("filtered: %d %d\n",current_point.x,current_point.y);
				//printf("raw: %d %d\n",current_raw_point.x,current_raw_point.y);
			}
			// plot cage
			cv::Point2i ftl,ftr,fbl,fbr,rtl,rtr,rbl,rbr;
			cv::Mat pos_ftr = (cv::Mat_<float>(4,1)<< (trajectory->points)[size-1].at<float>(0) - 0.25,(trajectory->points)[size-1].at<float>(1) + 0.3,trajectory->height,0);
			cv::Mat pos_fbl = (cv::Mat_<float>(4,1)<< (trajectory->points)[size-1].at<float>(0) - 0.25,(trajectory->points)[size-1].at<float>(1) - 0.3,0,0);
			cv::Mat pos_rtr = (cv::Mat_<float>(4,1)<< (trajectory->points)[size-1].at<float>(0) + 0.25,(trajectory->points)[size-1].at<float>(1) + 0.3,trajectory->height,0);
			cv::Mat pos_rbl = (cv::Mat_<float>(4,1)<< (trajectory->points)[size-1].at<float>(0) + 0.25,(trajectory->points)[size-1].at<float>(1) - 0.3,0,0);
			ftr = world_to_pixel( pos_ftr );
			fbl = world_to_pixel( pos_fbl );
			ftl = point_min(ftr,fbl);
			fbr = point_max(ftr,fbl);
			rtr = world_to_pixel( pos_rtr );
			rbl = world_to_pixel( pos_rbl );
			rtl = point_min(rtr,rbl);
			rbr = point_max(rtr,rbl);
			char pers[15];
			sprintf(pers,"Person %d", trajectory->id);
			cv::putText(current_image, std::string(pers), ftl, cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255,255,255), 2, 8, false);
			cv::rectangle(current_image,ftl,fbr, trajectory->color,1);
			cv::rectangle(current_image,rtl,rbr, trajectory->color,1);
			cv::line(current_image,ftl,rtl,trajectory->color,1);
			cv::line(current_image,ftr,rtr,trajectory->color,1);
			cv::line(current_image,fbl,rbl,trajectory->color,1);
			cv::line(current_image,fbr,rbr,trajectory->color,1);



			trajectory++;
		}
		char delay_s[100];
		sprintf(delay_s,"delay (min/max/avg): %4.1f/%4.1f/%4.1fms, log: %s vid: %s", min_delay,max_delay,avg_delay,(logging)?"on":"off",(video)?"on":"off");
		cv::addWeighted(lower_ribbon,0.5,ribbon_roi,0.5,0.,ribbon_roi);
		cv::putText(current_image, std::string(delay_s), cv::Point2i(5,470), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255,255,255), 1, 8, false);
		cv::addWeighted(overlay_roi, 0.5, overlay_image, 0.5, 0. , overlay_roi);
	}
	else{
		current_image = cv::Mat::zeros(480,640+240,CV_8UC3);
		cv::Mat camera_roi = current_image(cv::Rect(0,0,640,480));
		cv::Mat map_roi = current_image(cv::Rect(640,0,240,360));
		cv::Mat stat_roi = current_image(cv::Rect(640,360,240,120));
		cv::cvtColor(background, camera_roi, CV_GRAY2RGB);

		std::vector<trajectory>::iterator trajectory = people_trajectories.begin();
		while(trajectory != people_trajectories.end()){

			if(!(trajectory->active)){
				trajectory++;
				continue;
			}
			if( trajectory->last_update + 3000000 < last_image_frame){
				trajectory->active = false;
				trajectory++;
				continue;
			}

			// plot path
			cv::Point2i last_point, current_point;
			cv::Point2i last_raw_point, current_raw_point;
			cv::Point2i small_point, current_small_point;
			last_point = world_to_pixel((trajectory->points)[0]);
			last_raw_point = world_to_pixel((trajectory->raw_points)[0]);
			small_point = coord_to_point((trajectory->raw_points)[0]);
			int size = (trajectory->points).size();
			if(size > 1){
				for(int i = 1; i < size; i++){
					current_point = world_to_pixel((trajectory->points)[i]);
					current_raw_point = world_to_pixel((trajectory->raw_points)[i]);
					current_small_point = coord_to_point((trajectory->raw_points)[i]);
					//cv::circle(image,current_point,10, cv::Scalar(255),1);
					cv::line(camera_roi,last_point,current_point,trajectory->color,2);
					cv::line(map_roi,small_point,current_small_point,trajectory->color,1);
					//cv::line(camera_roi,last_raw_point,current_raw_point,trajectory->color,1);
					last_point = current_point;
					last_raw_point = current_raw_point;
					small_point = current_small_point;
				}
				cv::circle(camera_roi,current_point,5, trajectory->color,2);
				cv::circle(map_roi,current_small_point,3, trajectory->color,1);
			}
			// plot cage
			cv::Point2i ftl,ftr,fbl,fbr,rtl,rtr,rbl,rbr;
			cv::Mat pos_ftr = (cv::Mat_<float>(4,1)<< (trajectory->points)[size-1].at<float>(0) - 0.25,(trajectory->points)[size-1].at<float>(1) + 0.3,trajectory->height,0);
			cv::Mat pos_fbl = (cv::Mat_<float>(4,1)<< (trajectory->points)[size-1].at<float>(0) - 0.25,(trajectory->points)[size-1].at<float>(1) - 0.3,0,0);
			cv::Mat pos_rtr = (cv::Mat_<float>(4,1)<< (trajectory->points)[size-1].at<float>(0) + 0.25,(trajectory->points)[size-1].at<float>(1) + 0.3,trajectory->height,0);
			cv::Mat pos_rbl = (cv::Mat_<float>(4,1)<< (trajectory->points)[size-1].at<float>(0) + 0.25,(trajectory->points)[size-1].at<float>(1) - 0.3,0,0);
			ftr = world_to_pixel( pos_ftr );
			fbl = world_to_pixel( pos_fbl );
			ftl = point_min(ftr,fbl);
			fbr = point_max(ftr,fbl);
			rtr = world_to_pixel( pos_rtr );
			rbl = world_to_pixel( pos_rbl );
			rtl = point_min(rtr,rbl);
			rbr = point_max(rtr,rbl);
			char pers[15];
			sprintf(pers,"Person %d", trajectory->id);
			cv::putText(camera_roi, std::string(pers), ftl, cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255,255,255), 2, 8, false);
			cv::rectangle(camera_roi,ftl,fbr, trajectory->color,2);
			cv::rectangle(camera_roi,rtl,rbr, trajectory->color,2);
			cv::line(camera_roi,ftl,rtl,trajectory->color,2);
			cv::line(camera_roi,ftr,rtr,trajectory->color,2);
			cv::line(camera_roi,fbl,rbl,trajectory->color,2);
			cv::line(camera_roi,fbr,rbr,trajectory->color,2);


			trajectory++;
		}
		//stats
		char min_delay_s[50],max_delay_s[50],avg_delay_s[50],log_s[50],vid_s[50];
		sprintf(min_delay_s,"Min delay: %4.1fms", min_delay);
		sprintf(max_delay_s,"Max delay: %4.1fms", max_delay);
		sprintf(avg_delay_s,"Avg delay: %4.1fms", avg_delay);
		sprintf(log_s,"logging:   %s", (logging)?"on":"off");
		sprintf(vid_s,"recording: %s", (video)?"on":"off");
		cv::putText(stat_roi, std::string(min_delay_s), cv::Point2i(5,15), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255,255,255), 1, 8, false);
		cv::putText(stat_roi, std::string(max_delay_s), cv::Point2i(5,40), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255,255,255), 1, 8, false);
		cv::putText(stat_roi, std::string(avg_delay_s), cv::Point2i(5,65), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255,255,255), 1, 8, false);
		cv::putText(stat_roi, std::string(log_s), cv::Point2i(5,90), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255,255,255), 1, 8, false);
		cv::putText(stat_roi, std::string(vid_s), cv::Point2i(5,115), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255,255,255), 1, 8, false);
	}

}

void set_rot_trans_matrices(float pos_x, float pos_y, float pos_z, float ro, float pi, float ya){


	pos_pixhawk = (cv::Mat_<float>(4,1) <<
				pos_x,
				pos_y,
				pos_z,
				0
		);

	// pixels to camera
	C2P = (cv::Mat_<float>(4,4) <<
			mx/f, 0,0, principle_x,
			0, my/f,0, principle_y,
			0, 0, 1/f, 0,
			0, 0, 0, 1);

	//Camera to Body
	cv::Mat C2B = (cv::Mat_<float>(4,4) <<
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

}

static void
mavlink_handler (const lcm_recv_buf_t *rbuf, const char * channel,
		const mavconn_mavlink_msg_container_t* container, void * user)
{
	const mavlink_message_t* msg = getMAVLinkMsgPtr(container);
	//mavlink_message_t response;
	//lcm_t* lcm = static_cast<lcm_t*>(user);
	//printf("Received message #%d on channel \"%s\" (sys:%d|comp:%d):\n", msg->msgid, channel, msg->sysid, msg->compid);

	switch(msg->msgid)
	{
	case MAVLINK_MSG_ID_OBJECT_DETECTED:
		{
			mavlink_object_detected_t od_msg;
			mavlink_msg_object_detected_decode(msg, &od_msg);
			cv::Mat position = (cv::Mat_<float>(4,1)<<
					od_msg.position[0],
					od_msg.position[1],
					od_msg.position[2],
					1.
			);
			cv::Mat error_a = (cv::Mat_<float>(3,1)<<
					od_msg.vector_x[0],
					od_msg.vector_x[1],
					0
			);
			cv::Mat error_b = (cv::Mat_<float>(3,1)<<
					od_msg.vector_y[0],
					od_msg.vector_y[1],
					0
			);
			float persons_height = od_msg.vector_z[0];

			correlate_to_trajectory(od_msg.tracking_id, od_msg.timestamp, position, error_a, error_b, persons_height);
			//statistics
			GTimeVal gtime;
			g_get_current_time(&gtime);
			uint64_t delay_time = ((uint64_t)gtime.tv_sec)*G_USEC_PER_SEC + ((uint64_t)gtime.tv_usec);
			float delay = (float)(delay_time - od_msg.timestamp)/1000;
			last_delay = delay;
			min_delay = std::min(min_delay, delay);
			max_delay = std::max(max_delay, delay);
			sum_delay += delay;
			avg_delay = sum_delay / count_delay;
			count_delay++;

			break;
		}
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

	//copy image to background
	if (ImageClient.readMonoImage(msg, background)){
		float px,py,pz,ro,pi,ya;
		last_image_frame = PxSHMImageClient::getTimestamp(msg);
		PxSHMImageClient::getGroundTruth(msg, px, py, pz);
		pz = camera_height;
		px = 0;
		py = 0;

		PxSHMImageClient::getRollPitchYaw(msg, ro, pi, ya);
		img_height = background.rows;
		img_width = background.cols;

		set_rot_trans_matrices(px,py,pz,ro,pi,ya);
		plot_trajectories();

		//show image and in case of videocapture save frame
		if(video){
			video_writer.write(current_image);
		}

		cv::imshow("Tracker",current_image);
		int key = cv::waitKey(3);
		//printf("%d\n",key);
		if(key == 1048687){
			// o for overlay
			overlay = !overlay;
		}
		else if(key == 1048694){
			// v for video
			if(video){
				video_writer.~VideoWriter();
				video = false;
				printf("stop recording\n");
			}
			else{
				time_t frame_now = time(NULL);
				char vid_name[50];
				strftime(vid_name,50,"../vid/vid_%d%m%Y_%H:%M:%S.avi",localtime(&frame_now));
				if( video_writer.open(std::string(vid_name), CV_FOURCC('M','P','4','2'), 16., cv::Size(880,480), 1) ){
					video = true;
				}
				else{
					printf("failed to open video file\n");
				}
			}
		}
		else if(key == 1048684){
			// l for logging
			if(!logging){
				time_t now = time(NULL);
				char log_name[50];
				strftime(log_name,50,"../log/log_tracker_%d%m%Y_%H:%M:%S.csv",localtime(&now));
				log_file.open(std::string(log_name));
				if(!log_file.is_open()){
					printf("log file could not be created, no logging takes place\n");
					logging = false;
				}
				else{
					logging = true;
					printf("begin logging\n");
					log_file << "begin logging\n";
				}
			}
			else{
				log_file.close();
				logging = false;
				printf("end logging\n");
			}
		}
		else if(key == 1048689){
			// q for quit
			quit = true;
			exit(EXIT_SUCCESS);
		}
	}
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

static GOptionEntry entries[] =
{
		{ "sysid", 'a', 0, G_OPTION_ARG_INT, &sysid, "ID of this system, 1-255", "42"},
		{ "compid", 'c', 0, G_OPTION_ARG_INT, &compid, "ID of this component, 1-255", "112" },
		{ "verbose", 'v', 0, G_OPTION_ARG_NONE, &verbose, "Be verbose", (verbose) ? "true" : "false" },
		{ "overlay", 'o', 0, G_OPTION_ARG_NONE, &overlay, "overlay detection", (overlay) ? "true" : "false" },
		{ "logging", 'l', 0, G_OPTION_ARG_NONE, &logging, "Log to file", (logging) ? "true" : "false" },
		{ "debug", 'd', 0, G_OPTION_ARG_NONE, &debug, "Debug mode, changes behaviour", (debug) ? "true" : "false" },
		{ "config", 'g', 0, G_OPTION_ARG_STRING, config_file, "Filename of paramClient config file", "../conf/main.cfg"},
		{ NULL }
};

int main (int argc, char ** argv)
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

	lcm_t * lcm;

	lcm = lcm_create ("udpm://");
	if (!lcm)
		{
			fprintf(stderr, "# ERROR: Cannot initialize LCM.\n");
			exit(EXIT_FAILURE);
		}

	mavconn_mavlink_msg_container_t_subscription_t * comm_sub =
			mavconn_mavlink_msg_container_t_subscribe (lcm, MAVLINK_MAIN /*MAVLINK_IMAGES*/, &mavlink_handler, lcm);

	// Thread
	GThread* lcm_thread;
	GError* err;

	if( !g_thread_supported() )
	{
		g_thread_init(NULL);
		// Only initialize g thread if not already done
	}

	if( (lcm_thread = g_thread_create((GThreadFunc)lcm_wait, (void *)lcm, TRUE, &err)) == NULL)
	{
		printf("Thread create failed: %s!!\n", err->message );
		g_error_free ( err ) ;
	}

	ImageClient.init(true, PxSHM::CAMERA_FORWARD_LEFT);
	// Subscribe to MAVLink messages on the image channel
    mavconn_mavlink_msg_container_t_subscription_t* imgSub = mavconn_mavlink_msg_container_t_subscribe(lcm, MAVLINK_IMAGES, &image_handler, &ImageClient);
    fprintf(stderr, "# INFO: Image client ready, waiting for images..\n");

	paramClient = new MAVConnParamClient(getSystemID(), compid, lcm, config_file->str, verbose);
	update_parameters();

	signal(SIGINT, signalHandler);

	cv::namedWindow("Tracker");
	//cv::createButton("Overlay Toogle",callback_overlay,NULL,CV_RADIOBOX,0);

	// prepare for logging
	if(logging){
		time_t now = time(NULL);
		char log_name[50];
		strftime(log_name,50,"../log/log_tracker_%d%m%Y_%H:%M:%S.csv",localtime(&now));
		log_file.open(std::string(log_name));
		if(!log_file.is_open()){
			printf("log file could not be created, no logging takes place\n");
			logging = false;
		}
	}

	while (!quit)
	{
		update_parameters();
		usleep(1000000);
		//printf("Waited another second while still receiving data in parallel\n");

	}

	if(logging){
		log_file.close();
	}

	mavconn_mavlink_msg_container_t_unsubscribe(lcm, imgSub);
	mavconn_mavlink_msg_container_t_unsubscribe (lcm, comm_sub);
	lcm_destroy (lcm);
	g_thread_join(lcm_thread);
	return 0;
}

