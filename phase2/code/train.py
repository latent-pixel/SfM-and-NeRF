from utils.data_utils import FetchImageData


image_data = FetchImageData('phase2/data/lego/', split='train')
test = image_data.get_transform(0)
print(test)
test_img = image_data.get_image(0)
test_img.show('Test Image. Successful.')
