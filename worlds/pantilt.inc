
define pantiltcamera camera
(
  width 64
  height 64
  horizfov 60
  vertfov 40
  yaw 0
  pitch 0

  color	"black"
  size3 [ 0.1 0.07 0.05 ]
  pose [ 0.02 0 0 ]
  obstacle_return 0
  laser_return 0
  ranger_return 0  
)

#TODO make a fancier model
define fancypantiltcamer pantiltcamera
(
   blocks 1

  # bottom
  block[0].points 4
  block[0].point[0] [ -2.1 0.07 ]
  block[0].point[1] [ -2.1 -0.07 ]
  block[0].point[2] [  2.1 -0.07 ]
  block[0].point[3] [  2.1 0.07 ]
  block[0].z [0 0.2 ]


)
