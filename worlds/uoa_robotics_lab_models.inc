# 
# Desc : This file defines all the models needed for the uoa_robotics_lab.world
# Authors : Nick Wong, Jui Chun(Jimmy) Peng Hsu
# Date : 21 July 2008
#

define modelBase model
(
    obstacle_return 1
    laser_return 1
    ranger_return 1
)

define table modelBase
(

  # These corresponds to the maximum x y and z of the table in meters.
  size [1.54 0.80 0.725]

  # Top most piece of the table main surface
  block( 
  points 4
  point[3] [  1.000  -1.000 ] 
  point[2] [ -1.000  -1.000 ] 
  point[1] [ -1.000   1.000 ] 
  point[0] [  1.000   1.000 ] 
  z [ 0.979 1 ]
  )

  # Lower piece of the table main surface
  block (
  points 4
  point[3] [  0.960  -1.000 ] 
  point[2] [ -0.960  -1.000 ] 
  point[1] [ -0.960   1.000 ] 
  point[0] [  0.960   1.000 ] 
  z [ 0.959 0.979 ]
			)
  # Top right leg
  block (
  points 4
  point[3] [  0.960   0.925 ] 
  point[2] [  0.920   0.925 ] 
  point[1] [  0.920   1.000 ] 
  point[0] [  0.960   1.000 ] 
  z [ 0.0 0.959 ]
			)
  # Top left leg
  block (
  points 4
  point[3] [  -0.920   0.925 ] 
  point[2] [  -0.960   0.925 ] 
  point[1] [  -0.960   1.000 ] 
  point[0] [  -0.920   1.000 ] 
  z [ 0.0 0.959 ]
			)
  # Bottom right leg
  block (
  points 4
  point[3] [  0.960   -1.000 ] 
  point[2] [  0.920   -1.000 ] 
  point[1] [  0.920   -0.925 ] 
  point[0] [  0.960   -0.925 ] 
  z [ 0.0 0.959 ]
			)
  # Bottom left leg
  block (
  points 4
  point[3] [  -0.960   -0.925 ] 
  point[2] [  -0.920   -0.925 ] 
  point[1] [  -0.920   -1.000 ] 
  point[0] [  -0.960   -1.000 ] 
  z [ 0.0 0.959 ]
			)
  # Side support - back
  block (
  points 4
  point[3] [  0.960   0.925 ] 
  point[2] [ -0.960   0.925 ] 
  point[1] [ -0.960   1.000 ] 
  point[0] [  0.960   1.000 ] 
  z [ 0.221 0.262 ]
			)
  # Side support - right
  block (
  points 4
  point[3] [  0.960  -1.000 ] 
  point[2] [  0.920  -1.000 ] 
  point[1] [  0.920   1.000 ] 
  point[0] [  0.960   1.000 ] 
  z [ 0.221 0.262 ]
			)
  # Side support - left
  block (
  points 4
  point[3] [  -0.920   -1.000 ] 
  point[2] [  -0.960   -1.000 ] 
  point[1] [  -0.960    1.000 ] 
  point[0] [  -0.920    1.000 ] 
  z [ 0.221 0.262 ]
			)
  # PC cage bottom support
  block (
  points 4
  point[3] [  0.960   -1.000 ] 
  point[2] [  0.587   -1.000 ] 
  point[1] [  0.587    0.275 ] 
  point[0] [  0.960    0.275 ] 
  z [ 0.221 0.262 ]
			)
  # PC cage left back support pole
  block (
  points 4
  point[3] [  0.627    0.200 ] 
  point[2] [  0.587    0.200 ] 
  point[1] [  0.587    0.275 ] 
  point[0] [  0.627    0.275 ] 
  z [ 0.221 0.979 ]
			)
  # PC cage right back support pole
  block (
  points 4
  point[3] [  0.960    0.200 ] 
  point[2] [  0.920    0.200 ] 
  point[1] [  0.920    0.275 ] 
  point[0] [  0.960    0.275 ] 
  z [ 0.221 0.979 ]
			)
  # PC cage left front support pole
  block (
  points 4
  point[3] [  0.627    -1.000 ] 
  point[2] [  0.587    -1.000 ] 
  point[1] [  0.587    -0.925 ] 
  point[0] [  0.627    -0.925 ] 
  z [ 0.221 0.979 ]
			) 
)

define chair1 modelBase
(
  # These corresponds to the maximum x y and z of the chair1 in meters.
  size [0.450 0.450 0.870]

  # Back support of the chair
  block (
  points 4
  point[3] [  1.000  -1.000 ] 
  point[2] [ -1.000  -1.000 ] 
  point[1] [ -1.000   -0.867 ] 
  point[0] [  1.000   -0.867 ] 
  z [ 0.483 1.000 ]
			)
  # Seat of the chair
  block (
  points 4
  point[3] [  1.000  -1.000 ] 
  point[2] [ -1.000  -1.000 ] 
  point[1] [ -1.000   1.000 ] 
  point[0] [  1.000   1.000 ] 
  z [ 0.483 0.517 ]
			)
  # Lower stand of the chair
  block (
  points 4
  point[3] [  1.000  -1.000 ] 
  point[2] [ -1.000  -1.000 ] 
  point[1] [ -1.000   1.000 ] 
  point[0] [  1.000   1.000 ] 
  z [ 0.0 0.034 ]
			)
  # Middle stick of the chair
  block (
  points 4
  point[3] [  0.100  -0.100 ] 
  point[2] [ -0.100  -0.100 ] 
  point[1] [ -0.100   0.100 ] 
  point[0] [  0.100   0.100 ] 
  z [ 0.0 0.483 ]
			)
)

define chair2 modelBase
(
  # These corresponds to the maximum x y and z of the chair1 in meters.
  size [0.450 0.450 0.870]

  # Back support of the chair
  block (
  points 4
  point[3] [  1.000  -1.000 ] 
  point[2] [ -1.000  -1.000 ] 
  point[1] [ -1.000   -0.867 ] 
  point[0] [  1.000   -0.867 ] 
  z [ 0.483 1.000 ]
			)
  # Seat of the chair
  block (
  points 4
  point[3] [  1.000  -1.000 ] 
  point[2] [ -1.000  -1.000 ] 
  point[1] [ -1.000   1.000 ] 
  point[0] [  1.000   1.000 ] 
  z [ 0.483 0.517 ]
			)
  # Lower stand of the chair - right
  block (
  points 4
  point[3] [  1.000  -1.000 ] 
  point[2] [  0.867  -1.000 ] 
  point[1] [  0.867   1.000 ] 
  point[0] [  1.000   1.000 ] 
  z [ 0.0 0.034 ]
			)
  # Right front leg
  block (
  points 4
  point[3] [  1.000   -1.000 ] 
  point[2] [  0.867   -1.000 ] 
  point[1] [  0.867   -0.867 ] 
  point[0] [  1.000   -0.867 ] 
  z [ 0.0 0.483 ]
			)
  # Right back leg
  block (
  points 4
  point[3] [  1.000   0.867 ] 
  point[2] [  0.867   0.867 ] 
  point[1] [  0.867   1.000 ] 
  point[0] [  1.000   1.000 ] 
  z [ 0.0 0.483 ]
			)
  # Left front leg
  block (
  points 4
  point[3] [  -0.867   -1.000 ] 
  point[2] [  -1.000   -1.000 ] 
  point[1] [  -1.000   -0.867 ] 
  point[0] [  -0.867   -0.867 ] 
  z [ 0.0 0.483 ]
			)
  # Left back leg
  block (
  points 4
  point[3] [  -0.867   0.867 ] 
  point[2] [  -1.000   0.867 ] 
  point[1] [  -1.000   1.000 ] 
  point[0] [  -0.867   1.000 ] 
  z [ 0.0 0.483 ]
			)
  # Lower stand of the chair - left
  block (
  points 4
  point[3] [  -0.867  -1.000 ] 
  point[2] [  -1.000  -1.000 ] 
  point[1] [  -1.000   1.000 ] 
  point[0] [  -0.867   1.000 ] 
  z [ 0.0 0.034 ]
			)
)
define bigCabin modelBase
(
  # These corresponds to the maximum x y and z of the chair1 in meters.
  size [1.540 0.600 1.300]

  # Top plane
  block (
  points 4
  point[3] [  1.000  -1.000 ] 
  point[2] [ -1.000  -1.000 ] 
  point[1] [ -1.000   1.000 ] 
  point[0] [  1.000   1.000 ] 
  z [ 1.275 1.300 ]
			)
  block (
  points 4
  point[3] [ 0.974  -0.933 ] 
  point[2] [ -0.974  -0.933 ] 
  point[1] [ -0.974   0.933 ] 
  point[0] [ 0.974   0.933 ] 
  z [ 1.250 1.275 ]
			)
  block (
  points 4
  point[3] [ 0.974  -0.933 ] 
  point[2] [ 0.94  -0.933 ] 
  point[1] [ 0.94   0.933 ] 
  point[0] [ 0.974   0.933 ] 
  z [ 0.730 1.250 ]
			)
  block (
  points 4
  point[3] [ -0.94  0.933 ] 
  point[2] [ -0.94  -0.933 ] 
  point[1] [ -0.974   -0.933 ] 
  point[0] [ -0.974   0.933 ] 
  z [ 0.730 1.250 ]
			)
  block (
  points 4
  point[3] [ 0.94  0.85 ] 
  point[2] [ -0.94  0.85 ] 
  point[1] [ -0.94   0.933 ] 
  point[0] [ 0.94   0.933 ] 
  z [ 0.730 1.250 ]
			)
  block (
 points 4
 point[3] [ -0.162  -0.933 ] 
 point[2] [ -0.162  -0.85 ] 
 point[1] [ -0.128   -0.85 ] 
 point[0] [ -0.128   -0.933 ] 
 z [ 0.730 1.250 ]
			)
 block (
  points 4
 point[3] [ 0.128  -0.933 ] 
 point[2] [ 0.128  -0.85 ] 
 point[1] [ 0.162   -0.85 ] 
 point[0] [ 0.162   -0.933 ] 
 z [ 0.730 1.250 ]
		  )
  # Middle plane
  block (
  points 4
  point[3] [ 0.974  -0.933 ] 
  point[2] [ -0.974  -0.933 ] 
  point[1] [ -0.974   0.933 ] 
  point[0] [ 0.974   0.933 ] 
  z [ 0.705 0.730 ]
			)
# Side walls
  block (
 points 4
 point[3] [ -0.94  0.933 ] 
 point[2] [ -0.94  -0.933 ] 
 point[1] [ -0.974   -0.933 ] 
 point[0] [ -0.974   0.933 ] 
 z [ 0.105 0.705 ]
			)
  block (
  points 4
  point[3] [ 0.974  -0.933 ] 
  point[2] [ 0.94  -0.933 ] 
  point[1] [ 0.94   0.933 ] 
  point[0] [ 0.974   0.933 ] 
  z [ 0.105 0.705 ]
			)
# Beams
  block (
 points 4
 point[3] [ -0.162  -0.933 ] 
 point[2] [ -0.162  -0.85 ] 
 point[1] [ -0.128   -0.85 ] 
 point[0] [ -0.128   -0.933 ] 
 z [ 0.105 0.705 ]
			)
  block (
 points 4
 point[3] [ 0.128  -0.933 ] 
 point[2] [ 0.128  -0.85 ] 
 point[1] [ 0.162   -0.85 ] 
 point[0] [ 0.162   -0.933 ] 
 z [ 0.105 0.705 ]
			)
# Back wall
  block (
  points 4
  point[3] [ 0.94  0.85 ] 
  point[2] [ -0.94  0.85 ] 
  point[1] [ -0.94   0.933 ] 
  point[0] [ 0.94   0.933 ] 
  z [ 0.105 0.705 ]
			)
# Bottom plane

  block (
  points 4
  point[3] [ 0.974  -0.933 ] 
  point[2] [ -0.974  -0.933 ] 
  point[1] [ -0.974   0.933 ] 
  point[0] [ 0.974   0.933 ] 
  z [ 0.080 0.105 ]
			)
# Wheels
  block (
  points 4
  point[3] [ 0.948  -0.933 ] 
  point[2] [ 0.880  -0.760 ] 
  point[1] [ 0.905   -0.69 ] 
  point[0] [ 0.974   -0.867 ] 
  z [ 0.00 0.08 ]
			)
  block (
  points 4
  point[3] [ -0.948  -0.933 ] 
  point[2] [ -0.880  -0.760 ] 
  point[1] [ -0.905   -0.69 ] 
  point[0] [ -0.974   -0.867 ] 
  z [ 0.00 0.08 ]
			)
  block (
  points 4
  point[3] [ -0.948  0.933 ] 
  point[2] [ -0.880  0.760 ] 
  point[1] [ -0.905   0.69 ] 
  point[0] [ -0.974   0.867 ] 
  z [ 0.00 0.08 ]
			)
  block (
  points 4
  point[3] [ 0.948  0.933 ] 
  point[2] [ 0.880  0.760 ] 
  point[1] [ 0.905  0.69 ] 
  point[0] [ 0.974  0.867 ] 
  z [ 0.00 0.08 ]
			)
 )


