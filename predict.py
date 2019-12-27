import numpy as np
import matplotlib.pyplot as plt
import cv2

from deepposekit.models import load_model
from deepposekit.io import DataGenerator, VideoReader, VideoWriter

import tqdm

INPUT_VIDEO_PATH = 'bug_videos/2019-11-18_09-57-40.mp4'
OUTPUT_VIDEO_PATH = 'fly_posture.mp4'


def predict_on_video(predictions):
    # Visualize the data as a video
    data_generator = DataGenerator('annotation_set.h5')

    predictions = predictions[..., :2]
    predictions *= 2

    resized_shape = (data_generator.image_shape[0] * 2, data_generator.image_shape[1] * 2)
    cmap = plt.cm.hsv(np.linspace(0, 1, data_generator.keypoints_shape[0]))[:, :3][:, ::-1] * 255

    writer = VideoWriter(OUTPUT_VIDEO_PATH, (192 * 2, 192 * 2), 'mp4v', 30.0)
    reader = VideoReader(INPUT_VIDEO_PATH, batch_size=1)

    for frame, keypoints in tqdm.tqdm(zip(reader, predictions)):
        print(keypoints)
        frame = frame[0]
        frame = frame.copy()
        frame = cv2.resize(frame, resized_shape)
        for idx, node in enumerate(data_generator.graph):
            if node >= 0:
                pt1 = keypoints[idx]
                pt2 = keypoints[node]
                cv2.line(frame, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0, 0, 255), 2, cv2.LINE_AA)
        for idx, keypoint in enumerate(keypoints):
            keypoint = keypoint.astype(int)
            cv2.circle(frame, (keypoint[0], keypoint[1]), 5, tuple(cmap[idx]), -1, lineType=cv2.LINE_AA)

        cv2.imshow('frame', frame)
        writer.write(frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    writer.close()
    reader.close()


if __name__ == '__main__':
#
#     # Load the trained model
#     model = load_model('best_model_densenet.h5')
#
#     # Read the video to predict
#     reader = VideoReader(INPUT_VIDEO_PATH, batch_size=10, gray=True)
#     frames = reader[0]
#
#     # Make predictions for the full video
#     predictions = model.predict(reader, verbose=1)
#     reader.close()
#     np.save('predictions.npy', predictions)
#
#
#     # Write the predictions on video
#     predict_on_video(predictions)

    predictions = np.load('predictions.npy')
    predict_on_video(predictions)