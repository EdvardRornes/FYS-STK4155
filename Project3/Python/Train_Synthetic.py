from utils import * 
from NNs import * 

if __name__ == "__main__":
    #### Creating example of synthethic GW data:

    # Parameters
    N = 5000
    T = 50; event_length = (N//10, N//8)
    window_size = 10
    batch_size = 512
    learning_rate = 1e-2
    regularization_value = 1e-7
    gw_earlyboost = 1.4
    epochs = 100
    clip_value = 2 

    SNR = 5

    t = np.linspace(0, T, N)

    # Background noise
    X = (0.5*np.sin(90*t) - 0.5*np.cos(60*t)*np.sin(-5.*t) + 0.3*np.cos(30*t) + 0.05*np.sin(N/40*t))/SNR

    # Add a single synthetic GW event to each sample
    generator = GWSignalGenerator(signal_length=N)
    # y_i = np.zeros_like(x) # For no background signal tests
    event = generator.generate_random_events(1, event_length)
    generator.apply_events(X, event)
    y = np.array(generator.labels)

    ### RNN model
    rnn = RNN(1, [16, 16, 2], 1, Adam(learning_rate=learning_rate), activation="tanh", activation_out="sigmoid", lambda_reg=regularization_value,
              loss_function=DynamicallyWeightedLoss(initial_boost=gw_earlyboost, epochs=epochs))
    rnn.train(X.reshape(-1, 1), y.reshape(-1,1), epochs, batch_size, window_size, clip_value=clip_value)

    y_pred = rnn.predict(X.reshape(-1,1,1))
    print(np.shape(y_pred), np.shape(y.flatten()))
    plt.plot(t, X, label="data")
    plt.plot(t, y.flatten(), label="actual")
    plt.plot(t, y_pred[:,0], label="predicted")
    plt.legend()
    plt.show()