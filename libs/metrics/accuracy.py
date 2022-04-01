

def accuracy(prediction,target):
    return (prediction == target).float().mean()


