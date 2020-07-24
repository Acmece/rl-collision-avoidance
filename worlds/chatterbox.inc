# Define a model based on the Autonomy Lab Chatterbox robot
#
# Author: Richard T Vaughan (rtv), Ash Charles
# $Id: chatterbox.inc,v 1.5 2008-01-15 01:25:42 rtv Exp $

include "irobot.inc"
include "hokuyo.inc"

define cbir sensor
(
  # define the size of each transducer [xsize ysize zsize] in meters
  size [0.01 0.03 0.01 ] 
  # define the range bounds [min max]
  range [0 1.5]
	# define the angular field of view in degrees
	fov 20
	# define the number of samples spread over the fov
  samples 1
	# define the color that ranges are drawn in the gui
	color_rgba [ 0 1 0 0.2 ] 
)

define chatterbox create
(
 joules 100000 # todo
 joules_capacity 400000 # todo
 watts 100 # todo
 
# charging connection
 model
 (
	pose [0.100 0.000 -0.090 0.000]
	take_watts 1000.0 # todo
	size [0.15 0.1 0.03]
	color "yellow"
	obstacle_return 0
 )

 model( 
        block( points 6  
               point[0] [-3 -1.5 ]
               point[1] [-3 0.5 ]
               point[2] [-2 1.5 ]
               point[3] [+2 1.5 ]
               point[4] [+3 0.5 ]
               point[5] [+3 -1.5 ]
               z [0 0.05]
							 )

        pose [-0.03 0 0 90] 
        size [0.25 0.20 0.05 ] 
        color "gray" 
   )
 
 ranger
 (	
  cbir( pose [ 0.050  0.000  0 0]	) # front forward
  cbir( pose [ 0.050  0.050  0 30]	) # front left side 
  cbir( pose [ 0.000  0.050  0 90]	) # left side
  cbir( pose [-0.050  0.000 0 180]	) # back backwards
  cbir( pose [ 0.000 -0.050 0 -90]	) # right side
  cbir( pose [ 0.050 -0.050 0 -30]	) # front right side
 )
 
 fiducial
 (
	pose [0.040 0.000 -0.005 0.000]
	range_min 0.0
	range_max 2.5
	range_max_id 2.5	
	fov 60 
	watts 10 # todo
  )

 )


define laserchatterbox chatterbox
(
 hokuyolaser()
) 


# Desc: Charging station to work with the chatterbox model
# Author: Ash Charles
# Date: 26 Febuary 2009

define charger model
(  
  size [ 0.040 0.200 0.100 ]
  color "gray70"
  
  # base pad
  model(
	color "gray50"
	size [0.10 0.20 0.001]
	pose [-0.070 0 -0.1 0]
	obstacle_return 0
  )
  # IR beam blocker
#  model(
#	color "gray30"
#    size [ 0.005 0.014 0.010]
#    pose [-0.020 0.000 -0.015]
#  )
  # IR beacons: Red=Port=Left; Right=Starboard=Green; Purple=Omni
  model(
	color "red"
	size [0.010 0.010 0.010]
	pose [-0.016 0.010 -0.015 -10 ]
#	fiducial_return 4
	)
  model(
	color "green"
	size [0.010 0.010 0.010]
	pose [-0.016 -0.010 -0.015 10 ]
#	fiducial_return 5
	)
  model(
	color "purple"
	size [0.010 0.010 0.010]
    pose [-0.020 0.000 -0.015 0.000]
    fiducial_return 6
    )

  # Positive Terminal of Charging Block 
  model( 
    pose [ -0.040 0.000 -0.100 0 ]
    color "yellow"
    size [ 0.020 0.100 0.005 ]
    joules -1  # provides infinite energy
    give_watts 1000 
	obstacle_return 0
   )
  
  # Negative terminal of the charging block
#  model( 
#    pose [ -0.040 -0.040 -0.100 0 ]
#    color "yellow"
#    size [ 0.020 0.020 0.005 ]
#   )
)
