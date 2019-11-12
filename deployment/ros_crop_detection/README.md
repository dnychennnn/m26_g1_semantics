### image\_extractor

ROS node that subscribes to an RGB and NIR camera image topic and writes the received images to files. Proposed to be used to extract images from a bagfile.

#### Usage

```
roslaunch image_extractor image_extractor.launch rgb_image_topic:='/camera/jai/rgb/image' nir_image_topic:='/camera/jai/nir/image' bagfile_path:='test.bag' output_path:='/tmp/'
```

Service call to get the number of images extracted so far:

```
rosservice call /image_extractor_node/get_image_count
```
