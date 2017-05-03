import sys
import io

import numpy as np

from keras.preprocessing.image import Iterator
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array


from keras import backend as K

try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None


def load_img_bytes(image_data, grayscale=False, target_size=None):
    """Loads an image into PIL format.
    # Arguments
        image_data: io.BytesIO(byte_array) of image 
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = pil_image.open(image_data)
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    if target_size:
        hw_tuple = (target_size[1], target_size[0])
        if img.size != hw_tuple:
            img = img.resize(hw_tuple)
    return img

class ImageDataGeneratorEx(ImageDataGenerator): 
   
    def __init__(self,
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0.,
        width_shift_range=0.,
        height_shift_range=0.,
        shear_range=0.,
        zoom_range=0.,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=False,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None,
        data_format=None):

        ImageDataGenerator.__init__(self,
            featurewise_center, 
            samplewise_center,
            featurewise_std_normalization,
            samplewise_std_normalization,
            zca_whitening, rotation_range,
            width_shift_range,
            height_shift_range,
            shear_range, zoom_range,
            channel_shift_range,
            fill_mode, cval,
            horizontal_flip,
            vertical_flip, rescale,
            preprocessing_function,
            data_format)

    def flow_from_database(self,
        target_size=(256, 256), color_mode='rgb',
        classes=None, get_class_array=lambda: [],
        get_image_count_by_class=lambda cl: 0,
        get_image_ids_by_class=lambda cl: [],
        get_image_data_by_id=lambda image_id: [],
        class_mode='categorical',
        batch_size=32, shuffle=True, seed=None,
        save_to_dir=None, save_prefix='',
        save_format='jpeg', follow_links=False):

        return DatabaseIterator(
            self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, get_class_array=get_class_array,
            get_image_count_by_class=get_image_count_by_class,
            get_image_ids_by_class=get_image_ids_by_class,
            get_image_data_by_id=get_image_data_by_id,
            class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links)

class DatabaseIterator(Iterator):

    def __init__(self, image_data_generator,
        target_size=(256, 256), color_mode='rgb',
        classes=None, get_class_array=lambda: [],
        get_image_count_by_class=lambda cl: 0,
        get_image_ids_by_class=lambda cl: [],
        get_image_data_by_id=lambda image_id: [],
        class_mode='categorical',
        batch_size=32, shuffle=True, seed=None,
        data_format=None,
        save_to_dir=None, save_prefix='', save_format='jpeg',
        follow_links=False):

        if data_format is None:
            data_format = K.image_data_format()
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}

        self.get_image_data_by_id = get_image_data_by_id

        # first, count the number of samples and classes
        self.samples = 0

        if not classes:
            classes = get_class_array()

        self.num_class = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        for cl in classes:
            self.samples += get_image_count_by_class(cl)

        print('Found %d images belonging to %d classes.' % (self.samples, self.num_class))

        # second, build an index of the images in the different class subfolders
        self.image_ids = []
        self.classes = np.zeros((self.samples,), dtype='int32')

        i = 0
        for c in classes:
            for image_id in get_image_ids_by_class(c):
                self.classes[i] = self.class_indices[c]
                i += 1
                self.image_ids.append(image_id)

        super(DatabaseIterator, self).__init__(self.samples, batch_size, shuffle, seed)

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data

        for i, j in enumerate(index_array):
            image_id = self.image_ids[j]
            image_data = self.get_image_data_by_id(image_id)
            img = load_img_bytes(image_data, target_size=self.target_size)
            x = img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.num_class), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y
