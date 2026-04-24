std_msgs/Header header
float32 origin_x # Origin x is min_x, bottom left corner 
float32 origin_y # Origin y is min_y, bottom left corner
float32 size_x # Size in metric max_x - min_x
float32 size_y # Size in metric max_y - min_y
float32 resolution
uint32 voxel_num_x # num voxel in x, should be size_x / res
uint32 voxel_num_y # num voxel in y, should be size_y / res
float32[] data # Assume just flattened from 2D array, default by row
uint8 task_id # Task ID
