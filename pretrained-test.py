if __name__ == '__main__':
    import os.path
    import keras
    from keras.datasets import cifar10
    from cifar10vgg import cifar10vgg
    from cifar10alexnet import cifar10alexnet
    from cifar10resnet import cifar10resnet
    import numpy as np
    import csv
    import matplotlib.pyplot as plt
    from fastconv.fastconv import get_flush_count

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    # y_train = keras.utils.to_categorical(y_train, 10)
    y_test_c = keras.utils.to_categorical(y_test, 10)

    MODE_STANDARD = 0
    MODE_ELIM_8 = 1
    MODE_ELIM_5 = 113

    modtype = "resnet"
    load = True
    orig = False
    flush = MODE_STANDARD

    if modtype == "vgg":
        model = cifar10vgg(load=load, orig=orig, flush=flush)
    elif modtype == "alexnet":
        model = cifar10alexnet(load=load, orig=orig, flush=flush)
    elif modtype == "resnet":
        model = cifar10resnet(load=load, orig=orig, flush=flush)
    else:
        exit(1)

    if not load:
        model.train(x_train, y_train, x_test, y_test)

    if True:
        weights = []
        for l in model.model.layers:
            if hasattr(l, 'kernel'):
                weights.append(l.kernel.numpy().reshape((-1)))
            if hasattr(l, 'bias'):
                weights.append(l.bias.numpy().reshape((-1)))
        weight_arr = np.concatenate(weights)

        log_weights = np.ma.log2(np.abs(weight_arr)).compressed()  #.filled(-50)
        h = np.histogram(log_weights, np.arange(-54, 10, 4))

        hrows = np.vstack([h[1], np.concatenate([h[0], [0]])]).transpose()

        print(np.min(log_weights), np.max(log_weights))
        print("zeros:", len(weight_arr) - np.count_nonzero(weight_arr))
        print(np.count_nonzero(log_weights < -14) / len(log_weights))

        #with open("weight-hist-" + modtype + ".csv", "w", newline='') as csvfile:
        #    writer = csv.writer(csvfile)
        #    writer.writerow(["start", "count"])
        #    writer.writerows(hrows)

        # plt.hist(log_weights, bins=20, log=True)
        plt.yscale("log")
        plt.bar(h[1][:-1], h[0])
        plt.show()

        exit(0)

    predicted_x = model.predict(x_test)
    np.save("logits-" + modtype + "-" + str(flush), predicted_x)

    residuals = np.argmax(predicted_x, 1) != np.argmax(y_test_c, 1)

    loss = sum(residuals) / len(residuals)
    print("the validation 0/1 loss is: ", loss, " acc ", 1 - loss)

    print("flushes:", get_flush_count(clear=True))

