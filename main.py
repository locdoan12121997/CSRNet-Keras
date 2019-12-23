import os

import cv2
import numpy as np

from CSRNet import CSRNet
from area_controller import AreaController
from utils_imgproc import norm_by_imagenet
from matplotlib import pyplot as plt

desired_size_resize = (858, 480)
desired_size = (480, 858)
working_areas = [((145, 0), (480, 200)), ((36, 317), (160, 430))]
heat_maps = list()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model = CSRNet(input_shape=(None, None, 3))
model.load_weights('./weights/model.hdf5')

cap = cv2.VideoCapture('MOT16-04.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4',fourcc, 30.0, desired_size_resize)
acccumulate_out = cv2.VideoWriter('accumulate_output.mp4',fourcc, 30.0, desired_size_resize)
frame2 = 0
area_controller = AreaController(working_areas, desired_size)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        original_frame = cv2.resize(frame, desired_size_resize)
        frame = norm_by_imagenet(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB).astype(np.float32))
        heat_map = model.predict(np.expand_dims(frame, 0))[0,:,:,0]
        number_of_people = np.sum(heat_map)
        area_controller.run(heat_map)

        total_working_area_numpy = area_controller.get_total_working_area_numpy()
        heat_map = heat_map * total_working_area_numpy
        heat_map = heat_map / np.amax(heat_map) * 255
        heat_map = heat_map.astype(np.uint8)
        frame = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)
        frame = cv2.resize(frame, desired_size_resize)
        info = "People: " + str(round(number_of_people))

        frame2 += frame
        normalized_frame = cv2.applyColorMap(frame2*10, cv2.COLORMAP_JET)
        normalized_frame = cv2.medianBlur(normalized_frame, 11)

        cv2.putText(normalized_frame, text=info, org=(50, 70),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(255, 255, 255), thickness=2)

        visualized_frame = np.where(frame > (128, 0, 0), frame, original_frame)

        cv2.imshow('frame1', normalized_frame)

        # # write the flipped frame
        out.write(np.uint8(visualized_frame))
        acccumulate_out.write(np.uint8(normalized_frame))
        cv2.imshow('frame',visualized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

normalized_frame = normalized_frame.astype(np.float32)
fig, ax = plt.subplots()
plt.axis('off')
im = ax.imshow(cv2.cvtColor(np.uint8(normalized_frame), cv2.COLOR_BGR2RGB), cmap='jet')
fig.colorbar(im, orientation='horizontal')
plt.savefig('output.png')

area_controller.draw_people_count()
# Release everything if job is finished
cap.release()
out.release()
acccumulate_out.release()
cv2.destroyAllWindows()