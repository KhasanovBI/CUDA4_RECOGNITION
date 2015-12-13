import numpy as np
import caffe
from settings import *


def get_longest_text_size(words):
    max_size = (0, 0)
    for word in words:
        size = cv2.getTextSize(word, FONT_FACE, FONT_SCALE, THICKNESS)[0]
        if size[0] > max_size[0]:
            max_size = size
    return max_size


def main():
    capture = cv2.VideoCapture(0)
    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = caffe.Net(MODEL_FILE,
                    TRAINED_FILE,
                    caffe.TEST)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.load(MEAN_FILE).mean(1).mean(1))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))

    labels = np.loadtxt(LABEL_FILE, str, delimiter='\t')

    while True:
        # Capture frame-by-frame
        ret, img = capture.read()
        net.blobs['data'].data[...] = transformer.preprocess('data', img)
        out = net.forward()

        top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
        words = [label.split(None, 1)[1] for label in labels[top_k].tolist()]

        overlay = img.copy()
        longest_text_size = get_longest_text_size(words)
        rectangle_top = img.shape[0] - longest_text_size[1] * 10
        cv2.rectangle(img, (0, img.shape[0]), (longest_text_size[0] + 10, rectangle_top - 10), (0, 0, 0),
                      cv2.cv.CV_FILLED)
        cv2.addWeighted(overlay, OPACITY, img, 1 - OPACITY, 0, img)

        i = 0
        y0 = rectangle_top + longest_text_size[1]
        for word in words:
            y = y0 + i * 2 * longest_text_size[1]
            i += 1
            cv2.putText(img, word, (5, y), FONT_FACE, FONT_SCALE, FONT_COLOR)
        cv2.imshow('Result', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
