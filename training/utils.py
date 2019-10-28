from matplotlib import pyplot as plot


def visualize(image, mask):
    plot.figure(1)
    plot.subplot(211)
    plot.imshow(image)

    plot.subplot(212)
    plot.imshow(mask)
    plot.show() 