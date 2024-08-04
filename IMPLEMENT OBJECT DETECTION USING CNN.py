import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
(X_num, y_num), _ = tf.keras.datasets.mnist.load_data()
X_num = np.expand_dims(X_num, axis=-1).astype(np.float32) / 255.0
grid_size = 16 
def make_numbers(X, y):
    for _ in range(3):
        idx = np.random.randint(len(X_num))
        number = X_num[idx] @ (np.random.rand(1, 3) + 0.1)  
        kls = y_num[idx]
        px, py = np.random.randint(0, 100), np.random.randint(0, 100)
        mx, my = (px + 14) // grid_size, (py + 14) // grid_size
        channels = y[my][mx]
        if channels[0] > 0:
            channels[0] = 1.0
            channels[1] = px - (mx * grid_size) 
            channels[2] = py - (my * grid_size)
            channels[3] = 28.0 
            channels[4] = 28.0  
            channels[5 + kls] = 1.0
        X[py:py + 28, px:px + 28] += number
def make_data(size=64):
    X = np.zeros((size, 128, 128, 3), dtype=np.float32)
    y = np.zeros((size, 8, 8, 15), dtype=np.float32)
    for i in range(size):
        make_numbers(X[i], y[i])
    X = np.clip(X, 0.0, 1.0)
    return X, y
def get_color_by_probability(p):
    if p < 0.3:
        return (1., 0., 0.) 
    if p < 0.7:
        return (1., 1., 0.)  
    return (0., 1., 0.) 

def show_predict(X, y, threshold=0.1):
    X = X.copy()
    for mx in range(8):
        for my in range(8):
            channels = y[my][mx]
            prob, x1, y1, x2, y2 = channels[:5]
            if prob < threshold:
                continue
            color = get_color_by_probability(prob)
            px, py = (mx * grid_size) + x1, (my * grid_size) + y1
            cv2.rectangle(X, (int(px), int(py)), (int(px + x2), int(py + y2)), color, 1)
            cv2.rectangle(X, (int(px), int(py - 10)), (int(px + 12), int(py)), color, -1)
            kls = np.argmax(channels[5:])
            cv2.putText(X, f'{kls}', (int(px + 2), int(py - 2)), cv2.FONT_HERSHEY_PLAIN, 0.7, (0.0, 0.0, 0.0))
    plt.imshow(X)
    plt.axis('off')

X, y = make_data(size=1)
show_predict(X[0], y[0])
plt.show()
