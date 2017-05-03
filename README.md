# keras-image-ex

image_ex.py containg keras ImageDataGenerator extended class to allow loading images from other sources.

Using ImageDataGeneratorEx class one can load images sotred directly in a SQL database. I use it to load images from SQLite db because dealing with lots of images becomes very complicated.

The added function is flow_from_database . It's a copy of original flow_from_directory with the difference of having lamda functions as parameters that are being called whenever generator requires corresponding info like list of classes or list of image identifiers...
