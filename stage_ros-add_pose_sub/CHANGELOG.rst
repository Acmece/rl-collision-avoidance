^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package stage_ros
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.7.5 (2015-09-16)
------------------
* Removed all references to FLTK/Fluid and use the upstream CMake config file instead.
* Added ``reset_positions`` service to stage (adds dependency on ``std_srvs``).
* Contributors: Aurélien Ballier, Daniel Claes, Scott K Logan, William Woodall

1.7.4 (2015-03-04)
------------------
* Added missing -ldl flag on newer versions of Ubuntu
* Contributors: William Woodall

1.7.3 (2015-01-26)
------------------
* Split up ``fltk`` dep into ``libfltk-dev`` and ``fluid``, only ``run_depend``'ing on fluid.
* Now supports multiple robots with multiple sensors.
* Fixed a bug on systems that cannot populate FLTK_INCLUDE_DIRS.
* Updated topurg model from "laser" to "ranger".
* Added -u option to use name property of position models as its namespace instead of "robot_0", "robot_1", etc.
* Contributors: Gustavo Velasco Hernández, Gustavo Velasco-Hernández, Pablo Urcola, Wayne Chang, William Woodall

1.7.2 (2013-09-19)
------------------
* Changed default GUI window size to 600x400
* Added velocity to ground truth odometry
* Fixed tf (yaw component) for the base_link->camera transform.
* Fixed ground truth pose coordinate system

1.7.1 (2013-08-30)
------------------
* Fixing warnings
* Small fixes
* Added RGB+3D-sensor interface (Primesense/Kinect/Xtion).
  * Publishes CameraInfo, depth image, RGBA image, tf (takes world-file pantilt paremeter into account)
  * Supports the "old" configuration (laser+odom) as well as camera+odom, laser+camera+odom and odom-only.
  Fixed laser transform height (previously was hardcoded at 0.15, now it takes robot height into account).
* Introduced changes from https://github.com/rtv/Stage/issues/34 with some changes (does not require lasers to be present and works without cameras).

1.7.0 (2013-06-27 18:15:07 -0700)
---------------------------------
- Initial move over from old repository: https://code.ros.org/svn/ros-pkg/stacks/stage
- Catkinized
- Stage itself is released as a third party package now
- Had to disable velocities in the output odometry as Stage no longer implements it internally.
- Updated rostest
- Updated rviz configurations
