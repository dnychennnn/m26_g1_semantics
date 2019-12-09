# Message wrapping around the ouput of our crop detection network.
# Need to dicuss is this is needed

std_msgs/Header header

sensor_msgs/Image input_image

sensor_msgs/Image semantic_class_labels
sensor_msgs/Image semantic_class_confidence

sensor_msgs/Image stem_keypoint_confidence
sensor_msgs/Image stem_offset_x
sensor_msgs/Image stem_offset_y

geometry_msgs/Point[] stem_positions
