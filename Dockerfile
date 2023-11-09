FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

# Utilities.
RUN apt update ; apt install -y nano git cmake wget htop software-properties-common
RUN DEBIAN_FRONTEND=noninteractive TZ="America/New_York" apt install -y admesh

# Install python packages.
RUN git clone https://github.com/ondrejbiza/shapewarping.git

# Install OpenSCAD.
RUN add-apt-repository --yes ppa:openscad/releases
RUN apt install -y openscad

# Install an old version of v-hacd. Works well for convex decomposition for collision checking.
COPY /v-hacd /workspace/v-hacd/
WORKDIR /workspace/v-hacd/src
RUN cmake -DCMAKE_BUILD_TYPE=Release CMakeLists.txt
RUN cmake --build .
RUN cp test/testVHACD /usr/bin/
WORKDIR /workspace

# # Allow pybullet to connect to the host's display.
# # Run `xhost +` on the host machine.
# # docker run --net=host --env="DISPLAY" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" -it image
# RUN apt install -y libglu1-mesa-dev libgl1-mesa-dri freeglut3-dev mesa-common-dev
# # Insert your nvidia driver version here.
# RUN apt install -y libnvidia-gl-510
# RUN apt install -y x11-apps

# # Allow USB connection to a realsense camera.
# RUN apt -y install libusb-1.0-0-dev
# RUN apt -y install libglib2.0-0

# # OMPL (https://github.com/ompl)
# RUN apt install -y libboost-all-dev
# RUN apt install -y libeigen3-dev
# # OMPL needs a very particular version of castxml ¯\_(ツ)_/¯
# RUN wget https://data.kitware.com/api/v1/item/5f6c9d2450a41e3d19a8e7e8/download
# RUN tar xf download
# RUN rm download
# ENV PATH /workspace/castxml/bin:${PATH}
# RUN pip install pygccxml pyplusplus
# RUN git clone https://github.com/ompl/ompl.git
# RUN mkdir -p /workspace/ompl/build/Release
# WORKDIR /workspace/ompl/build/Release
# RUN cmake ../.. -DPYTHON_EXEC=/opt/conda/bin/python
# RUN make -j 8 update_bindings
# RUN make -j 8
# RUN make install
# WORKDIR /workspace