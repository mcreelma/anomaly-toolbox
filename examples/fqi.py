from functools import partial
from glob import glob
from pathlib import Path
from typing import Tuple, Union

import tensorflow as tf
import tensorflow_datasets as tfds

# import the characteristcs from the anomaly detection dataset class 
from anomaly_toolbox.datasets.dataset import AnomalyDetectionDataset


class FQI(AnomalyDetectionDataset):
    """FQI dataset."""

    # def __init__(self, path: Path):
    def __init__(self, path = 'fqi/'): # initialize with the path to the dataset
        super().__init__() # call the init of the parent class
        self._path = Path(path)  # Convert the path to a Path object
        self._channels = 1  # Assuming grayscale images (i.e., one channel)


    """Configure the dataset."""
    def configure( # configure the dataset
        self, 
        batch_size: int, # batch size is an integer 
        new_size: Tuple[int, int], # new size is a tuple of integers
        anomalous_label: Union[int, str, None] = None,  # anomalous label is an integer, string or None
        class_label: Union[int, str, None] = None, # the class label is an integer, string or None
        shuffle_buffer_size: int = 10000, # shuffle buffer size is an integer
        cache: bool = True, # chach the extra images (?)
        drop_remainder: bool = True, # drop the remainder of the images (?)
        output_range: Tuple[float, float] = (-1.0, 1.0), # what should the expected values for the dataset be
    ) -> None:

        def _read_and_map_fn(label):
            def fn(filename):

                binary = tf.io.read_file(str(filename))

                try:
                    # image = tf.image.decode_and_crop_jpeg(binary, 
                    #                                     #   channels=self._channels ,
                    #                                       channels=1,
                    #                                       crop_window=[0, 0, 512, 512])
                    # image = tf.image.decode_jpeg(binary, channels=self._channels)
                    image = tf.image.decode_fits(binary, name=None, hdu=0, memmap=False)
                    
                    # Print the shape of the image tensor
                    print("Image shape:", tf.shape(image))

                except tf.errors.InvalidArgumentError:

                    raise ValueError("Invalid or corrupted image: {}".format(filename))

                # Check if the image tensor has a valid shape
                if image.shape.ndims is None:
                    
                    raise ValueError("Invalid shape for image tensor.")
                

                return image, label

            return fn
        
        pipeline = partial(
            self.pipeline,
            new_size=new_size,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            cache=cache,
            drop_remainder=drop_remainder,
            output_range=output_range,
        )

        pipeline_train = partial(pipeline, is_training=True)
        pipeline_test = partial(pipeline, is_training=False)

        is_anomalous = lambda _, label: tf.equal(label, anomalous_label)
        is_normal = lambda _, label: tf.not_equal(label, anomalous_label)

        glob_ext = "*.fits.gz"  # Modify the extension accordingly
        normpath = f"{self._path}/Good/{glob_ext}"

        ## Checks 
        import os
        print('CWD: ' , os.getcwd())

        print('Filepath: ' , normpath)
        ############################################


        all_normal = glob(normpath)
        all_normal_train = all_normal[:10000]

        if not all_normal:
            raise ValueError("No normal images found.")
        # ...

        all_anomalous = glob(f"{self._path}/Bad/{glob_ext}")
        all_anomalous_train = all_anomalous[:10000]

        if not all_anomalous:
            raise ValueError("No anomalous images found.")

        # ...

        self._train_anomalous = tf.data.Dataset.from_tensor_slices(all_anomalous_train).map(
            _read_and_map_fn(self.anomalous_label)
        ).apply(pipeline_train)

        self._train_normal = tf.data.Dataset.from_tensor_slices(all_normal_train).map(
            _read_and_map_fn(self.normal_label)
        ).apply(pipeline_train)

        if not self._train_anomalous:
            raise ValueError("No anomalous images in the training dataset.")

        if not self._train_normal:
            raise ValueError("No normal images in the training dataset.")


        self._train = self._train_anomalous.concatenate(self._train_normal)

        # ...
