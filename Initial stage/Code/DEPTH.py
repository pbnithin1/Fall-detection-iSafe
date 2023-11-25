import pyrealsense2 as rs

# Create a context object
ctx = rs.context()

# Get the list of devices
devices = ctx.devices()

# Select the L515 device
l515 = devices[0]

# Create a pipeline object
pipeline = rs.pipeline()

# Add the L515 device to the pipeline
pipeline.add_device(l515)

# Start the pipeline
pipeline.start()

# Get the depth data
depth_data = l515.get_depth()

# Print the depth data
print(depth_data)
