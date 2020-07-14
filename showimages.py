def displayims(X,row,col):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(20,(row/col)*12))
    for x in range(row*col):
        plt.subplot(row,col,x+1)
        plt.imshow(X[x,:,:,0])
    plt.show()
