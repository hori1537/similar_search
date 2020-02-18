import keras

from keras.applications.vgg19 import VGG19

base_model = VGG19(weights="imagenet")
base_model.summary()

model = Model(
  inputs=base_model.input,
  outputs=base_model.get_layer("fc2").output
)


from annoy import AnnoyIndex

dim = 4096
annoy_model = AnnoyIndex(dim)

from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input

#directory_path = r"C:\Users\yuki\Desktop\python\similar\data"
#test_img_path = r"C:\Users\yuki\Desktop\python\similar\i_35812.png"

directory_path = "./data"
test_img_path = "test.png"


test_img = image.load_img(test_img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

fc2_features = model.predict(x)
annoy_model.add_item(i, fc2_features[0])

import glob
png_list = glob.glob(directory_path + "/*.png")

for png_path in png_list:

    img = image.load_img(png_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    fc2_features = model.predict(x)
    annoy_model.add_item(png_path, fc2_features[0])


annoy_model.build(332)
annoy_model.save("images_vgg19.ann")

items = trained_model.get_nns_by_item(2, 3, search_k=-1, include_distances=False)
print(items)

img = image.load_img(test_img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
fc2_features = model.predict(x)

result = trained_model.get_nns_by_vector(fc2_features[0], 3, search_k=-1, include_distances=False)
print(result)
