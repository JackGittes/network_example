import test
import params as pm
import data_generator as dg

img, lbs = dg.load_mnist(path='MNIST/test',kind='t10k')
w = pm.weight_loader('params/', weight_names='weight_100')
acc = test.test_on(img, lbs, w, im_len=len(img))
print('the total accuracy is:', str(100 * acc) + '%')