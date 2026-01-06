import numpy as np
from sklearn.datasets import fetch_openml

def load_mnist_data():
    print("loading mnist dataset ...")
    
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    images = mnist.data.astype('float32')
    labels = mnist.target.astype('int64')
    
    x_train = images[:60000]
    y_train = labels[:60000]
    
    x_test = images[60000:]
    y_test = labels[60000:]
    
    return x_train, y_train, x_test, y_test

def save_matrix_to_file(filename, x_data, y_data):
    print(f"saving {filename} ...")
    y_reshaped = y_data.reshape(-1, 1).astype('float32')
    combined = np.hstack((y_reshaped, x_data))
    np.savetxt(filename, combined, fmt='%.6g')


x_train, y_train, x_test, y_test = load_mnist_data()

print(f"training data shape: {x_train.shape}")
print(f"test data shape: {x_test.shape}")

save_matrix_to_file('mnist_train.txt', x_train, y_train)
save_matrix_to_file('mnist_test.txt', x_test, y_test)