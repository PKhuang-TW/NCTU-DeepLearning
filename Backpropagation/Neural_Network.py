import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)


class Neural_Network():
    def __init__(self, X, y, hiddenSize):
        self.inputNum = len(X)  # Number of input
        self.inputSize = len(X[0])  # Input node size
        self.outputSize = len(y[0])  # Output node size
        self.hiddenSize = hiddenSize  # Size of each hidden layer
        
        self.lr = 0.5  # Initial learning rate
        self.loss = []
        
        # Initial weight
        self.w = []
        self.w.append(np.random.randn(self.inputSize, self.hiddenSize[0]))  # input -> hidden #1
        for idx in range(len(self.hiddenSize)-1):  # hidden #idx -> hidden #idx+1
            self.w.append(np.random.randn(self.hiddenSize[idx], self.hiddenSize[idx+1]))
        self.w.append(np.random.randn(self.hiddenSize[-1], self.outputSize))  # last hidden -> output
        
        # Initial i/o, delta of each layer
        self.L_in = []
        self.L_out = []
        self.delta = []
        self.L_in.append(X)  # Set L_in[0] as original input
        self.L_out.append(X/np.amax(X, axis=0))  # L_out[0] normalize the value of orignial input
        self.delta.append(np.array([None]))  # The first layer has no delta
        for idx in range(len(self.hiddenSize)):
            # delta : applying derivative of sigmoid to error (inputNum * hiddenSize)
            self.delta.append( np.array( [[None]*self.hiddenSize[idx]]*self.inputNum ))
            self.L_in.append ( np.array( [[None]*self.hiddenSize[idx]]*self.inputNum ))
            self.L_out.append(np.array( [[None]*self.hiddenSize[idx]]*self.inputNum ))
        # delta : applying derivative of sigmoid to error (inputNum * outputSize)
        self.L_in.append ( np.array( [[None]*self.outputSize]*self.inputNum ))
        self.L_out.append(np.array( [[None]*self.outputSize]*self.inputNum ))
        self.delta.append( np.array( [[None]*self.outputSize]*self.inputNum ))
        
    def forward(self, y):
        for idx in range(1, len(self.L_out)):
            # Each col represents value of each node (input_num * hidden_size)
            self.L_in[idx] = np.matmul(self.L_out[idx-1], self.w[idx-1])
            self.L_out[idx] = self.sigmoid(self.L_in[idx])
        
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoid_prime(self, x):
        return x * (1 - x)

    def backward(self, y):
        # delta : applying derivative of sigmoid to error (inputNum * LayerSize)
        self.delta[-1] = (y-self.L_out[-1]) * self.sigmoid_prime(self.L_out[-1])
        for idx in reversed(range( 1, len(self.L_out)-1 )):
            self.delta[idx] = np.matmul(self.delta[idx+1], self.w[idx].T)
            self.delta[idx] *= self.sigmoid_prime(self.L_out[idx])
            
        # adjusting weights (input_size * h1_size)
        for idx in range(len(self.w)):
            self.w[idx] += self.lr * np.matmul( self.L_out[idx].T, self.delta[idx+1] )
            
    # MSE
    def cal_loss(self, y, o):
        error = o - y
        return np.sum(np.square(error)) / len(y)

    def train(self, X, y, epoch):
        self.forward(y)
        self.loss.append(self.cal_loss(y, self.L_out[-1]))
        # If current loss does not change obviously,
        # then divide learning rate by 2
        if (epoch+1) % 5000 == 0:
            print('epoch %s loss : %s' %(str(epoch+1), str(self.loss[-1])))
            if 0.9 < self.loss[-1]/self.loss[-4999] < 1.1:
                self.lr /= 2
        self.backward(y)

    def predict(self, input, y):
        self.L_in[0] = input  # Set L_in[0] as original input
        self.L_out[0] = (input/np.amax(input, axis=0))  # L_out[0] normalize the value of orignial input
        self.forward(y)
        print(self.L_out[-1])


def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)


def generate_XOR_easy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        if 0.1*i == 0.5:
            continue
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21, 1)


def show_result(x, y, pred_y):
    plt.subplot(1,2,1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] <= 0.5:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] <= 0.5:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.show()
    

def show_loss(epoch, loss):
    plt.plot(range(epoch+1), loss)
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


def main():
	# Linear data
	epoch = 100000
	X, y = generate_linear(100)
	Xpredict = X

	hiddenSize = (10,5,3,5)
	print('\nStart Training Neural Network...\n')
	NN = Neural_Network(X, y, hiddenSize)
	for epoch in range(epoch):
	    NN.train(X, y, epoch)
	show_result(X, y, NN.L_out[-1])
	show_loss(epoch, NN.loss)

	# Print Linear Prediction
	print('\n####################\n\nLinear Data Prediction :')
	NN.predict(Xpredict, y)
	print('\n####################\n\nPress enter to continue...')
	while(input()==0):
		continue


	# XOR data
	epoch = 100000
	X, y = generate_XOR_easy()
	Xpredict = X

	hiddenSize = (10, 8, 6, 4)
	print('\nStart Training Neural Network...\n')
	NN = Neural_Network(X, y, hiddenSize)
	for epoch in range(epoch):
	    NN.train(X, y, epoch)
	show_result(X, y, NN.L_out[-1])
	show_loss(epoch, NN.loss)

	# Print XOR Prediction
	print('\n####################\n\nXOR Data Prediction :')
	NN.predict(Xpredict, y)


if __name__ == '__main__':
	main()