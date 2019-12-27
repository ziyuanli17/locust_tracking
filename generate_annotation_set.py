import numpy as np
import matplotlib.pyplot as plt
import tqdm
import pandas as pd
from deepposekit.io import VideoReader, DataGenerator, initialize_dataset
from deepposekit.annotate import KMeansSampler

INPUT_VIDEO_PATH = 'bug_videos/2019-11-18_09-57-40.mp4'


# Sample video frames
def sample_frames(reader):
    randomly_frames = []
    for idx in tqdm.tqdm(range(len(reader) - 1)):
        # read frames
        batch = reader[idx]
        # sample random frames
        random_sample = batch[np.random.choice(batch.shape[0], 10, replace=False)]
        randomly_frames.append(random_sample)
    reader.close()

    # Apply k-means to reduce correlation
    kmeans = KMeansSampler(n_clusters=10, max_iter=1000, n_init=10, batch_size=100, verbose=True)
    kmeans.fit(np.concatenate(randomly_frames))
    kmeans.plot_centers(n_rows=2)
    plt.show()

    return kmeans.sample_data(np.concatenate(randomly_frames), n_samples_per_label=10)


if __name__ == "__main__":
    skeleton = pd.read_csv('skeleton.csv')
    reader = VideoReader(INPUT_VIDEO_PATH, batch_size=100, gray=True)

    kmeans_sampled_frames, kmeans_cluster_labels = sample_frames(reader)
    print(kmeans_sampled_frames.shape)

    # Initialize a new data set for annotations
    initialize_dataset(
        images=kmeans_sampled_frames,
        datapath='annotation_set.h5',
        skeleton='skeleton.csv',
    )

    # Create a data generator
    data_generator = DataGenerator('annotation_set.h5', mode="full")
    image, keypoints = data_generator[0]

    plt.figure(figsize=(5, 5))
    image = image[0] if image.shape[-1] is 3 else image[0, ..., 0]
    cmap = None if image.shape[-1] is 3 else 'gray'
    plt.imshow(image, cmap=cmap, interpolation='none')
    for idx, jdx in enumerate(data_generator.graph):
        if jdx > -1:
            plt.plot(
                [keypoints[0, idx, 0], keypoints[0, jdx, 0]],
                [keypoints[0, idx, 1], keypoints[0, jdx, 1]],
                'r-'
            )
    # Set image-keypoints pair to zero
    plt.scatter(keypoints[0, :, 0], keypoints[0, :, 1], c=np.arange(data_generator.keypoints_shape[0]), s=50, cmap=plt.cm.hsv, zorder=3)
    plt.show()

