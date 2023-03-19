#!/bin/bash
fd_model="face-detection-0204"
lm_model="landmarks-regression-retail-0009"
id_model="face-reidentification-retail-0095"

fd_device="GPU"
lm_device="GPU"
id_device="GPU"

path_face_images="/home/installer/faces"

python tuan_recognition.py -i 4 \
    -m_fd  "/home/installer/intel/${fd_model}/FP16/${fd_model}.xml" \
    -m_lm "/home/installer/intel/${lm_model}/FP16/${lm_model}.xml" \
    -m_reid "/home/installer/intel/${id_model}/FP16/${id_model}.xml" \
    -d_fd ${fd_device} \
    -d_lm ${lm_device} \
    -d_reid ${id_device} \
    -fg ${path_face_images}
