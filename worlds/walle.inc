
# Desc: A garbage collecting robot
# Author: Richard Vaughan
# Date: 21 Nov 2008
# CVS: $Id: pioneer.inc,v 1.30 2008-01-15 01:25:42 rtv Exp $


define walle position 
(
  color "gray50"
  drive "diff"		 	# Differential steering model.

  obstacle_return 1           	# Can hit things.
  laser_return 1                 # reflects laser beams
  ranger_return 1             	# reflects sonar beams
  blobfinder_return 1         	# Seen by blobfinders  
  fiducial_return 1           	# Seen as "1" fiducial finders

  localization "gps"         	  

  size [0.35 0.5 0.5]
  #origin [-0.25 -0.25 0 0 ]

  # body bottom
  block 
  (
    points 4
    point[0] [ 0.1 0.25 ] 
    point[1] [ 0.9 0.25 ] 
    point[2] [ 0.9 0.75 ] 
    point[3] [ 0.1 0.75 ] 
    z [ 0.2 0.6 ]
  )

  # body top
  block 
  (
    points 4
    point[0] [ 0.1 0.25 ] 
    point[1] [ 0.9 0.25 ] 
    point[2] [ 0.9 0.75 ] 
    point[3] [ 0.1 0.75 ] 
    z [ 0.6 0.7 ]
  )

  # right wheel 
  block 
  (
    points 4
    point[0] [ 1 0 ] 
    point[1] [ 1 0.25 ] 
    point[2] [ 0.0 0.25 ] 
    point[3] [ 0 0.0 ] 
    z [ 0.0 0.4 ]
  )

  # right wheel 
  block 
  (
    points 4
    point[0] [ 1 0.75 ] 
    point[1] [ 1 1 ] 
    point[2] [ 0 1 ] 
    point[3] [ 0 0.75 ] 
    z [ 0.0 0.4 ]
  )

  # right eye front
  block
  (
	 points 4
	 point[0] [0.8 0.30]
	 point[1] [0.8 0.49]
	 point[2] [0.6 0.49]
	 point[3] [0.6 0.30]
	 z [0.8 0.95]
  )

  # left eye front
  block
  (
	 points 4
	 point[0] [0.8 0.51]
	 point[1] [0.8 0.7]
	 point[2] [0.6 0.7]
	 point[3] [0.6 0.51]
	 z [0.8 0.95]
  )

  # right eye rear
  block
  (
	 points 4
	 point[0] [0.6 0.34]
	 point[1] [0.6 0.49]
	 point[2] [0.25 0.49]
	 point[3] [0.25 0.34]
	 z [0.82 0.93]
  )

  # neck
  block (
   points 4
	 point[3] [0.3 0.45]
	 point[2] [0.3 0.55]
	 point[1] [0.4 0.55]
	 point[0] [0.4 0.45]
	 z [0.7 0.9]
  )   

  # right eye rear
  block
  (
	 points 4
	 point[0] [0.6 0.51]
	 point[1] [0.6 0.66]
	 point[2] [0.25 0.66]
	 point[3] [0.25 0.51]
	 z [0.82 0.93]
  )

  # right arm
  block
  (
	 points 4
	 point[0] [1.1 0.18]
	 point[1] [1.1 0.25]
	 point[2] [0.35 0.25]
	 point[3] [0.35 0.18]
	 z [0.55 0.65]
  )

  # left arm
  block
  (
	 points 4
	 point[0] [0.35 0.82]
	 point[1] [0.35 0.75]
	 point[2] [1.1 0.75]
	 point[3] [1.1 0.82]
	 z [0.55 0.65]
  )

  # left hand right finger
  block
  (
	 points 4
	 point[0] [1.1 0.78]
	 point[1] [1.1 0.70]
	 point[2] [1.2 0.70]
	 point[3] [1.2 0.78]
	 z [0.5 0.65]
  )

  # left hand left finger
  block
  (
	 points 4
	 point[0] [1.1 0.87]
	 point[1] [1.1 0.79]
	 point[2] [1.2 0.79]
	 point[3] [1.2 0.87]
	 z [0.5 0.65]
  )

  # right hand right finger
  block
  (
	 points 4
	 point[0] [1.1 0.21]
	 point[1] [1.1 0.13]
	 point[2] [1.2 0.13]
	 point[3] [1.2 0.21]
	 z [0.5 0.65]
  )

  # right hand left finger
  block
  (
	 points 4
	 point[0] [1.1 0.30]
	 point[1] [1.1 0.22]
	 point[2] [1.2 0.22]
	 point[3] [1.2 0.30]
	 z [0.5 0.65]
  )

)
