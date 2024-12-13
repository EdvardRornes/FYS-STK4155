from utils import * 
# from NNs import * 
from test2 import RNN 

if __name__ == "__main__":
    #### Creating example of synthethic GW data:

    # Parameters
    N = 1_000
    T = 50; event_length = (N//10, N//8)
    event_length_test = (N//10, N//8)

    window_size = 10
    batch_size = N
    learning_rate = 1e-2
    regularization_value = 1e-7
    gw_earlyboost = 1
    epochs = 100
    clip_value = 5

    SNR = 25_000

    t = np.linspace(0, T, N)

    # Background noise
    X1 = (0.5*np.sin(90*t) - 0.5*np.cos(60*t)*np.sin(-5.*t) + 0.3*np.cos(30*t) + 0.05*np.sin(N/40*t))/SNR
    X2 = (0.5*np.sin(40*t) - 0.5*np.cos(25*t)*np.sin(-10*t) + 0.3*np.cos(60*t) + 0.10*np.sin(N/18*t))/SNR
    X_test = (0.5*np.sin(50*t) - 0.5*np.cos(80*t)*np.sin(-10*t) + 0.3*np.cos(40*t) + 0.05*np.sin(N/20*t))/SNR

    # Add a single synthetic GW event to each sample
    generator = GWSignalGenerator(signal_length=N)
    # y_i = np.zeros_like(x) # For no background signal tests
    event = generator.generate_random_events(1, event_length)
    generator.apply_events(X1, event)
    y1 = np.array(generator.labels)

    generator = GWSignalGenerator(signal_length=N)
    # y_i = np.zeros_like(x) # For no background signal tests
    event = generator.generate_random_events(1, event_length)
    generator.apply_events(X2, event)
    y2 = np.array(generator.labels)

    generator = GWSignalGenerator(signal_length=N)
    event = generator.generate_random_events(1, event_length)
    generator.apply_events(X_test, event)
    y_test = np.array(generator.labels)

    ### RNN model
    rnn = RNN(1, [16, 16, 2], 1, Adam(learning_rate=learning_rate), activation="tanh", activation_out="sigmoid", lambda_reg=regularization_value,
              loss_function=DynamicallyWeightedLoss(initial_boost=gw_earlyboost, epochs=epochs), scaler="standard")
    
    X = [X1.reshape(-1,1), X2.reshape(-1,1)]
    y = [y1.reshape(-1,1), y2.reshape(-1,1)]
    rnn.train_multiple_data(X, y, epochs, 128, window_size, clip_value=clip_value)

    y_pred = rnn.predict(X_test, y_test, window_size)
    plt.plot(t, X_test, label="data")
    plt.plot(t, y_test.flatten(), label="actual")
    t_pred = t[:len(t) - (window_size-1)]
    t_pred = [t[i + window_size // 2] for i in range(0, len(t) - window_size + 1, window_size)]

    t_pred = []
    for i in range(0, len(t) - window_size + 1, 1):
        seq = t[i:i + window_size]
        middle = seq[-1]
        
        t_pred.append(middle)



    plt.plot(t, Activation.sigmoid(X_test))
    plt.plot(t_pred, y_pred[:,0], label="predicted")
    plt.legend()
    plt.show()