/*
 *  stageros
 *  Copyright (c) 2008, Willow Garage, Inc.
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

/**

@mainpage

@htmlinclude manifest.html
**/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <signal.h>


// libstage
#include <stage.hh>


// roscpp
#include <ros/ros.h>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h>

#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Twist.h>
#include <rosgraph_msgs/Clock.h>

#include <std_srvs/Empty.h>

#include "tf/transform_broadcaster.h"

#define USAGE "stageros <worldfile>"
#define STAGE "stage"
#define IMAGE "image"
#define DEPTH "depth"
#define CAMERA_INFO "camera_info"
#define ODOM "odom"
#define BASE_SCAN "scan"
#define BASE_POSE_GROUND_TRUTH "base_pose_ground_truth"
#define CMD_VEL "cmd_vel"

#define POSE "cmd_pose"
#define IS_CRASHED "is_crashed"

// Our node
class StageNode
{
private:
  
  // roscpp-related bookkeeping
  ros::NodeHandle n_;
  
  // A mutex to lock access to fields that are used in message callbacks
  boost::mutex msg_lock;
  
  struct Ranger
  {
    StageNode* node; // this
    ros::Publisher scan_pub;
    // todo - pub pose on parent
  };
  
  struct Position
  {
    StageNode* node; // this
    ros::Publisher odom_pub;
    ros::Publisher ground_truth_pub;
    ros::Subscriber cmdvel_sub; 
    ros::Subscriber pose_sub;

    ros::Publisher stall_pub;

  };
  
  
  // Used to remember initial poses for soft reset
  std::vector<Stg::ModelPosition *> positionmodels;  
  std::vector<Stg::Pose> initial_poses;
  ros::ServiceServer reset_srv_;
  
  ros::Publisher clock_pub_;
  
  bool isDepthCanonical;
  
  // A helper function that is executed for each stage model.  We use it
  // to search for models of interest.
  static void s_import_models(Stg::Model* mod, StageNode* node);
  
  static bool s_update_world(Stg::World* world, StageNode* node){
    node->WorldCallback();
    return false; // thank you, call again
  }
  
  static bool s_update_ranger( Stg::ModelRanger* mod, Ranger* r ){
    r->node->RangerCallback( mod, r );
    return false; // thankyou, call again
  } 

  static bool s_update_position( Stg::ModelPosition* mod, Position* p ){
    p->node->PositionCallback( mod, p );    
    return false; // thankyou, call again
  } 

  // create a topic name from a model name and topic type
  const char *mapName(const char *name, Stg::Model* mod) const;
  
  tf::TransformBroadcaster tf;
  
  // Last time that we received a velocity command
  ros::Time base_last_cmd;
  ros::Duration base_watchdog_timeout;
  
  // Current simulation time
  ros::Time sim_time;
  
  // Last time we saved global position (for velocity calculation).
  ros::Time base_last_globalpos_time;
  // Last published global pose of each robot
  std::vector<Stg::Pose> base_last_globalpos;

  int position_model_idx=0;
  
public:
  // Constructor; stage itself needs argc/argv.  fname is the .world file
  // that stage should load.
  StageNode(int argc, char** argv, bool gui, const char* fname );
  ~StageNode();
  
  void ImportModel(Stg::Model* mod );
  
  // these are called by Stage when the world and models are updated
  void WorldCallback();  
  void RangerCallback( Stg::ModelRanger* mod, Ranger* r );
  void PositionCallback( Stg::ModelPosition* mod, Position* p );
  
  // Do one update of the world.  May pause if the next update time
  // has not yet arrived.
  bool UpdateWorld();
  
  // Message callback for a MsgBaseVel message, which set velocities.
  void cmdvelReceived(int idx, const boost::shared_ptr<geometry_msgs::Twist const>& msg);
  
  // Message callback for a cmd_pose message, which sets positions.
  void poseReceived(int idx, const boost::shared_ptr<geometry_msgs::Pose const>& msg);

  // Service callback for soft reset
  bool cb_reset_srv(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response);
  
  // The main simulator object
  Stg::World* world;
};


const char *
StageNode::mapName(const char *name, Stg::Model* mod) const
{
  // NOTE static buffer safe in single-thread code onlyx
  static char buf[256];
  snprintf(buf, sizeof(buf), "/%s/%s/%s", STAGE, mod->Token(), name );
    
  // replace colons with underscores, and dots with slashes
  for( char* c = buf; *c; ++c )
    {
      if( *c == ':' ) *c = '_';
      if( *c == '.' ) *c = '/';
    }
    
  return buf;
}


void
StageNode::ImportModel(Stg::Model* mod )
{
  boost::mutex::scoped_lock lock(msg_lock);

  if (dynamic_cast<Stg::ModelRanger *>(mod)) {
        
    Stg::ModelRanger* mr = dynamic_cast<Stg::ModelRanger *>(mod);
    assert(mr);
    
    Ranger* r = new Ranger;
    assert( r );    
    r->node = this;
    r->scan_pub = n_.advertise<sensor_msgs::LaserScan>(mapName(BASE_SCAN, mr), 10); // TODO - replace 0 with ranger number

    mr->AddCallback( Stg::Model::CB_UPDATE,
		     (Stg::model_callback_t)s_update_ranger,
		     (void*)r );

    mr->Subscribe(); // TODO: wait until someone needs the data
  }
  else if (dynamic_cast<Stg::ModelPosition *>(mod)) {
    Stg::ModelPosition * mp = dynamic_cast<Stg::ModelPosition *>(mod);
      // remember initial poses
    positionmodels.push_back(mp);
      initial_poses.push_back(mp->GetGlobalPose());

      Position* p = new Position;
      assert(p);
      p->node = this;

      p->odom_pub = n_.advertise<nav_msgs::Odometry>(mapName(ODOM, mp), 10);
      p->stall_pub = n_.advertise<std_msgs::Int8>(mapName(IS_CRASHED, mp), 10);
 
      p->ground_truth_pub = n_.advertise<nav_msgs::Odometry>(mapName(BASE_POSE_GROUND_TRUTH, mp), 10);
      p->cmdvel_sub = n_.subscribe<geometry_msgs::Twist>(mapName(CMD_VEL, mp), 10, boost::bind(&StageNode::cmdvelReceived, this, position_model_idx, _1));
      p->pose_sub = n_.subscribe<geometry_msgs::Pose>(mapName(POSE, mp), 10, boost::bind(&StageNode::poseReceived, this, position_model_idx, _1));

      mp->AddCallback( Stg::Model::CB_UPDATE,
		       (Stg::model_callback_t)s_update_position,
		       (void*)p );

      mp->Subscribe(); // TODO: wait until someone needs the data
      position_model_idx+=1;
    }
  else if (dynamic_cast<Stg::ModelCamera *>(mod)) {
    ROS_WARN( "STAGEROS WARN: Camera models/topics not currently supported" );
  }
}

void
StageNode::s_import_models(Stg::Model* mod, StageNode* node)
{
  node->ImportModel( mod );
}

bool
StageNode::cb_reset_srv(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response)
{
  ROS_INFO("Resetting stage!");
  for (size_t r = 0; r < this->positionmodels.size(); r++) {
    this->positionmodels[r]->SetPose(this->initial_poses[r]);
    this->positionmodels[r]->SetStall(false);
  }
  return true;
}



void
StageNode::cmdvelReceived(int idx, const boost::shared_ptr<geometry_msgs::Twist const>& msg)
{
    boost::mutex::scoped_lock lock(msg_lock);
    this->positionmodels[idx]->SetSpeed(msg->linear.x,
                                        msg->linear.y,
                                        msg->angular.z);
    this->base_last_cmd = this->sim_time;
}

void
StageNode::poseReceived(int idx, const boost::shared_ptr<geometry_msgs::Pose const>& msg)
{
    boost::mutex::scoped_lock lock(msg_lock);
    Stg::Pose pose;

    double roll, pitch, yaw;
    tf::Matrix3x3 m(tf::Quaternion(msg->orientation.x,msg->orientation.y,msg->orientation.z,msg->orientation.w));
    m.getRPY(roll, pitch, yaw);
    pose.x = msg->position.x;
    pose.y = msg->position.y;
    pose.z = 0;
    pose.a = yaw;
    this->positionmodels[idx]->SetPose(pose);
}


StageNode::StageNode(int argc, char** argv, bool gui, const char* fname )
{
    this->sim_time.fromSec(0.0);
    this->base_last_cmd.fromSec(0.0);
    double t;
    ros::NodeHandle localn("~");
    if(!localn.getParam("base_watchdog_timeout", t))
        t = 0.2;
    this->base_watchdog_timeout.fromSec(t);

    if(!localn.getParam("is_depth_canonical", isDepthCanonical))
        isDepthCanonical = true;

    // We'll check the existence of the world file, because libstage doesn't
    // expose its failure to open it.  Could go further with checks (e.g., is
    // it readable by this user).
    struct stat s;
    if(stat(fname, &s) != 0)
    {
        ROS_FATAL("The world file %s does not exist.", fname);
        ROS_BREAK();
    }

    // initialize libstage
    Stg::Init( &argc, &argv );

    if(gui)
        this->world = new Stg::WorldGui(600, 400, "Stage (ROS)");
    else
        this->world = new Stg::World();

    this->world->Load(fname);

    // install update callbacks and subscribe to every model
    // (todo: on demand subscriptions)
    this->world->ForEachDescendant((Stg::model_callback_t)s_import_models, this);

    // every time the world is updated, we publish the sim_time
    this->world->AddUpdateCallback((Stg::world_callback_t)s_update_world, this);

    n_.setParam("/use_sim_time", true);	
    clock_pub_ = n_.advertise<rosgraph_msgs::Clock>("/clock", 10);

    // advertising reset service
    reset_srv_ = n_.advertiseService("reset_positions", &StageNode::cb_reset_srv, this);
}

StageNode::~StageNode()
{    
  // we leak all the installed callback structures here.
  // this is not a problem right now, since this is destroyed only at exit.
}

bool
StageNode::UpdateWorld()
{
    return this->world->UpdateAll();
}

void StageNode::RangerCallback( Stg::ModelRanger* mod, Ranger* r )
{
  assert( r );
  assert( mod );
  
  boost::mutex::scoped_lock lock(msg_lock);
  
  this->sim_time.fromSec(world->SimTimeNow() / 1e6);
  // We're not allowed to publish clock==0, because it used as a special
  // value in parts of ROS, #4027.
  if(this->sim_time.sec == 0 && this->sim_time.nsec == 0)
    {
      ROS_DEBUG("Ranger skipping initial simulation step, to avoid publishing clock==0");
      return;
    }
  
  const std::vector<Stg::ModelRanger::Sensor>& sensors = mod->GetSensors();
  
  if( sensors.size() > 1 && this->base_watchdog_timeout.toSec() == 0 )
    ROS_WARN( "ROS Stage currently supports rangers with 1 sensor only." );
  
  // for now we access only the zeroth sensor of the ranger - good
  // enough for most laser models that have a single beam origin
  const Stg::ModelRanger::Sensor& sensor = sensors[0];
  
  if( sensor.ranges.size() )
    {
      // Translate into ROS message format and publish
      sensor_msgs::LaserScan msg;
      msg.angle_min = -sensor.fov/2.0;
      msg.angle_max = +sensor.fov/2.0;
      msg.angle_increment = sensor.fov/(double)(sensor.sample_count-1);
      msg.range_min = sensor.range.min;
      msg.range_max = sensor.range.max;
      msg.ranges.resize(sensor.ranges.size());
      msg.intensities.resize(sensor.intensities.size());
      
      for(unsigned int i = 0; i < sensor.ranges.size(); i++)
	{
	  msg.ranges[i] = sensor.ranges[i];
	  msg.intensities[i] = sensor.intensities[i];
	}
      
      msg.header.frame_id = mapName("base_laser_link", mod->Parent() ); // todo replace 0 with ranger number
      
      msg.header.stamp = sim_time;
      r->scan_pub.publish(msg);
    }
  
  // Also publish the base->base_laser_link Tx.  This could eventually move
  // into being retrieved from the param server as a static Tx.
  Stg::Pose lp = mod->GetPose();
  tf::Quaternion laserQ;
  laserQ.setRPY(0.0, 0.0, lp.a);
  tf::Transform txLaser =  tf::Transform(laserQ, tf::Point(lp.x, lp.y, mod->Parent()->GetGeom().size.z + lp.z));
  
  tf.sendTransform(tf::StampedTransform(txLaser, sim_time,
					mapName("base_link", mod->Parent() ),
					mapName("base_laser_link", mod->Parent())));
}

void StageNode::PositionCallback( Stg::ModelPosition* mod, Position* p )
{
  assert( p );
  assert( mod );
  
  boost::mutex::scoped_lock lock(msg_lock);
  
  this->sim_time.fromSec(world->SimTimeNow() / 1e6);
  // We're not allowed to publish clock==0, because it used as a special
  // value in parts of ROS, #4027.
  if(this->sim_time.sec == 0 && this->sim_time.nsec == 0)
    {
      ROS_DEBUG("Ranger skipping initial simulation step, to avoid publishing clock==0");
      return;
    }

    // TODO make this only affect one robot if necessary
  if((this->base_watchdog_timeout.toSec() > 0.0) &&
     ((this->sim_time - this->base_last_cmd) >= this->base_watchdog_timeout))
    {
      mod->SetSpeed(0.0, 0.0, 0.0);
    }
  
  //the position of the robot
  tf.sendTransform(tf::StampedTransform(tf::Transform::getIdentity(),
					sim_time,
					mapName("base_footprint", mod),
					mapName("base_link", mod)) );
  
  // Get latest odometry data
  // Translate into ROS message format and publish
  nav_msgs::Odometry odom_msg;
  odom_msg.pose.pose.position.x = mod->est_pose.x;
  odom_msg.pose.pose.position.y = mod->est_pose.y;
  odom_msg.pose.pose.orientation = tf::createQuaternionMsgFromYaw(mod->est_pose.a);
  Stg::Velocity v = mod->GetVelocity();
  odom_msg.twist.twist.linear.x = v.x;
  odom_msg.twist.twist.linear.y = v.y;
  odom_msg.twist.twist.angular.z = v.a;
  
  //@todo Publish stall on a separate topic when one becomes available
  //this->odomMsgs[r].stall = this->positionmodels[r]->Stall();
  //
  odom_msg.header.frame_id = mapName("odom", mod);
  odom_msg.header.stamp = sim_time;
  
  p->odom_pub.publish(odom_msg);
    
  //Publish stall on separate topic
  ROS_INFO("Stalled status: %d", p->positionmodel->Stalled());
  std_msgs::Int8 stall_status;
  stall_status.data = p->positionmodel->Stalled();
  p->stall_pub.publish(stall_status);

  // broadcast odometry transform
  tf::Quaternion odomQ;
  tf::quaternionMsgToTF(odom_msg.pose.pose.orientation, odomQ);
  tf::Transform txOdom(odomQ, tf::Point(odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, 0.0));
  tf.sendTransform(tf::StampedTransform(txOdom, sim_time,
					mapName("odom", mod),
					mapName("base_footprint", mod)));
  
  // Also publish the ground truth pose and velocity
  Stg::Pose gpose = mod->GetGlobalPose();
  tf::Quaternion q_gpose;
  q_gpose.setRPY(0.0, 0.0, gpose.a);
  tf::Transform gt(q_gpose, tf::Point(gpose.x, gpose.y, 0.0));

  // Velocity is 0 by default and will be set only if there is previous pose and time delta>0
  Stg::Velocity gvel(0,0,0,0);
  if (this->base_last_globalpos.size()>1){ // TODO 1?x
    Stg::Pose prevpose = this->base_last_globalpos.at(0);
    double dT = (this->sim_time-this->base_last_globalpos_time).toSec();
    if (dT>0)
      gvel = Stg::Velocity(
			   (gpose.x - prevpose.x)/dT,
			   (gpose.y - prevpose.y)/dT,
			   (gpose.z - prevpose.z)/dT,
			   Stg::normalize(gpose.a - prevpose.a)/dT
			   );
    this->base_last_globalpos.at(0) = gpose;
  }else //There are no previous readings, adding current pose...
    this->base_last_globalpos.push_back(gpose);
  
  nav_msgs::Odometry ground_truth_msg;
  ground_truth_msg.pose.pose.position.x     = gt.getOrigin().x();
  ground_truth_msg.pose.pose.position.y     = gt.getOrigin().y();
  ground_truth_msg.pose.pose.position.z     = gt.getOrigin().z();
  ground_truth_msg.pose.pose.orientation.x  = gt.getRotation().x();
  ground_truth_msg.pose.pose.orientation.y  = gt.getRotation().y();
  ground_truth_msg.pose.pose.orientation.z  = gt.getRotation().z();
  ground_truth_msg.pose.pose.orientation.w  = gt.getRotation().w();
  ground_truth_msg.twist.twist.linear.x = gvel.x;
  ground_truth_msg.twist.twist.linear.y = gvel.y;
  ground_truth_msg.twist.twist.linear.z = gvel.z;
  ground_truth_msg.twist.twist.angular.z = gvel.a;
  
  ground_truth_msg.header.frame_id = mapName("odom", mod);
  ground_truth_msg.header.stamp = sim_time;
  
  p->ground_truth_pub.publish(ground_truth_msg);
}


void
StageNode::WorldCallback()
{
  if( ! ros::ok() ) {
    ROS_INFO( "ros::ok() is false. Quitting." );
    this->world->QuitAll();
    return;
  }
  
  boost::mutex::scoped_lock lock(msg_lock);

  this->sim_time.fromSec(world->SimTimeNow() / 1e6);
  // We're not allowed to publish clock==0, because it used as a special
  // value in parts of ROS, #4027.
  if(this->sim_time.sec == 0 && this->sim_time.nsec == 0)
    {
      ROS_DEBUG("World skipping initial simulation step, to avoid publishing clock==0");
      return;
    }
  
  this->base_last_globalpos_time = this->sim_time;
  rosgraph_msgs::Clock clock_msg;
  clock_msg.clock = sim_time;
  this->clock_pub_.publish(clock_msg);
}

int 
main(int argc, char** argv)
{
    if( argc < 2 )
    {
        puts(USAGE);
        exit(-1);
    }

    ros::init(argc, argv, "stageros");

    bool gui = true;

    for(int i=0;i<(argc-1);i++)
    {
        if(!strcmp(argv[i], "-g"))
            gui = false;
    }

    StageNode sn(argc-1,argv,gui,argv[argc-1] );

    boost::thread t = boost::thread(boost::bind(&ros::spin));

    sn.world->Start();

    Stg::World::Run();
    
    t.join();

    exit(0);
}

