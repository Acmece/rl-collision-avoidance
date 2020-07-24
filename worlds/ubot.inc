# device definitions for the UMASS UBot

# the ubot has an IR array
define ubot_ir reb_ir
(
  add_noise 1
 # noiseparams [1.913005560938 -7.728130591833] # used to generate noise

  power_on 1
  min_range 0.10 # in meters 
  max_range 0.8  # in meters

  ircount 8
  irpose[0] [ 0.035	0.0	0.0 ]
  irpose[1] [ 0.025	0.025	45 ]
  irpose[2] [ 0.0	0.035	90 ]
  irpose[3] [ -0.025	0.025	135 ]
  irpose[4] [ -0.035	0.0	180 ]
  irpose[5] [ -0.025	-0.025	225 ]
  irpose[6] [ 0.0	-0.035	270 ]
  irpose[7] [ 0.025	-0.025	315 ]
)

define ubot reb_position
(
  size [ 0.19	0.19 0.19]
  offset [ 0.0	0.0 ]
  ubot_ir()
)
