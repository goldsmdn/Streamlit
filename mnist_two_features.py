import streamlit as st

st.write("# MNIST dimensionality reduction")
st.write('A key concept in **Machine Learning** is reducing the dimensionality of objects to the key features.')
st.write('This is an example with MNIST digits, 28*28 pixel hand written digits')
st.write("In this example MNIST digits are reconstructed used a Keras autoencoder with only two features in the latent space.")
st.write("The impact of changing the feature values and running the decoder to reconstruct the images can be found")
st.image("autoencoder.png", caption="Keras MNIST autoencoder")

st.write("Importing modules...")

from keras.datasets import mnist

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Input
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D, Flatten, Dense, Reshape
from keras.models import Model
from keras.models import load_model
from keras import optimizers

import numpy as np

import matplotlib.pyplot as plt

LATENT_DIMS = 2
SAMPLES = 100

st.write("Loading MNIST data...")

(train_X, train_Y), (test_X, test_Y) = mnist.load_data()
label_dict = {
    0: 'Zero',
    1: 'One',
    2: 'Two',
    3: 'Three',
    4: 'Four',
    5: 'Five',
    6: 'Six',
    7: 'Seven',
    8: 'Eight',
    9: 'Nine',
}
train_X = train_X.reshape(-1, 28, 28, 1)
test_X = test_X.reshape(-1, 28, 28, 1)
train_X = train_X.astype('float32')/255.
test_X = test_X.astype('float32')/255.
train_X, valid_X, train_ground, valid_ground = train_test_split(train_X,
                                                                train_X,
                                                            test_size=0.2,
                                                            random_state=13)
if 'confirmed' not in st.session_state:
    st.session_state.confirmed = False

if 'model_ready' not in st.session_state:
    st.session_state.model_ready = False

if 'parameters_confirmed' not in st.session_state:
    st.session_state.parameters_confirmed = False

retrain_text = st.selectbox(
                             "Do you want to retrain the model - if you do this will take a long time and won't add much value!",
                             ("False", "True")
                             )

# Button to confirm selection
if st.button("Confirm Selection"):
    st.session_state.confirmed = True

if st.session_state.confirmed == True:
    if retrain_text == "True":
        retrain = True
    elif retrain_text == "False":
        retrain = False
    else:
        st.write("Please choose False or True")

    if retrain: 
        st.write("Please enter and confirm the parameters")

        batch_size = st.number_input("Enter the batch size",
                                    min_value=0, 
                                    max_value=512, 
                                    value=128,
                                    step=1,
                                    )
        
        epochs = int(st.number_input("Enter the number of epochs",
                            min_value=0, 
                            max_value=50, 
                            value=3,
                            step=1,
                            ))

        # Button to confirm selection
        if st.button("Confirm Selection for training parameters"):
            st.session_state.parameters_confirmed = True
            if st.session_state.parameters_confirmed == True:
                st.write("Retraining data - this may take several minutes")
                input_img = Input(shape=(28, 28, 1))
                conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)  # 28 x 28 x 32
                pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 14 x 14 x 32
                conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)  # 14 x 14 x 64
                pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 7 x 7 x 64
                conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)  # 7 x 7 x 128
                bottleneck1 = Flatten()(conv3)
                encoded = Dense(LATENT_DIMS, activation='relu', name='bottleneck_layer')(bottleneck1)
                encoder = Model(input_img, encoded)

                decoder_input= Input(shape=(LATENT_DIMS,))
                decoder_dense = Dense(49, activation='relu')(decoder_input)
                d = Reshape((7, 7, 1))(decoder_dense)
                conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(d)  # 7 x 7 x 128
                up1 = UpSampling2D((2, 2))(conv4)  # 14 x 14 x 128
                conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)  # 14 x 14 x 64
                up2 = UpSampling2D((2, 2))(conv5)  # 28 x 28 x 64
                decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2)  # 28 x 28 x 1
                decoder = Model(decoder_input, decoded)

                encoded = encoder(input_img)
                decoded = decoder(encoded)

                autoencoder = Model(input_img, decoded)
                autoencoder.compile(loss='mean_squared_error', optimizer=optimizers.RMSprop())

                autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size, epochs=epochs, verbose=2,
                                                    validation_data=(valid_X, valid_ground))

                loss = autoencoder_train.history['loss']
                val_loss = autoencoder_train.history['val_loss']
                epochs = range(epochs)

                st.write("### Training results:")
                fig1, ax1 = plt.subplots()

                ax1.plot(epochs, loss, 'bo', label='Training loss')
                ax1.plot(epochs, val_loss, 'b', label='Validation loss')
                ax1.set_title('Training and validation loss')
                ax1.legend()
                st.pyplot(fig1)
                encoder.save("encoder.keras")
                decoder.save("decoder.keras")
                st.session_state.model_ready = True
    else:
        st.session_state.model_ready = True
    
    if st.session_state.model_ready == True:
        decoder = load_model("decoder.keras")
        encoder = load_model("encoder.keras")
        col1, col2 = st.columns(2, vertical_alignment="center")

        dim1 = st.slider("Adjust the slider to change the value of the **first** feature in the reconstructed image",
                          0, 25, 3, key="dim1")
        dim2 = st.slider("Adjust the slider to change the value of the **second** feature in the reconstructed image", 
                         0, 25, 3, key="dim2")
        curr_label = f'Reconstructed image with feature 1={dim1}, feature 2={dim2}'
        my_input = np.array([[dim1, dim2]])
        my_output = decoder.predict(my_input)

        bottleneck_prediction = encoder.predict(test_X[:SAMPLES])
        labels = list(label_dict.values())
        colors = plt.cm.get_cmap('tab10', len(labels))  # Get a colormap with enough colors

        # Create the plot
        plt.figure(figsize=(4, 4)) # Adjust figure size for better visualization

        fig2, ax2 = plt.subplots()

        for i in range(len(labels)):
            plt.scatter(bottleneck_prediction[test_Y[:SAMPLES] == i, 0],
                        bottleneck_prediction[test_Y[:SAMPLES] == i, 1],
                        color=colors(i),
                        label=labels[i],
                        alpha=0.7) # Add alpha for better visibility with overlapping points

        ax2.set_title('The MNIST numbers with two reduced dimensions')
        ax2.set_xlabel("Feature 1") # Add x-axis label
        ax2.set_ylabel("Feature 2") # Add y-axis label
        ax2.legend()
        ax2.grid(True) # Add grid for better readability
        with col1:
            st.header("Scatter graph")
            st.pyplot(fig2)

        with col2:
            st.header("Reconstructed digits")
            plt.figure(figsize=(3, 3))
            fig3, ax3 = plt.subplots()
            ax3.axis('off') # Turn off axes for the subplot
            ax3.set_title(curr_label)
            ax3.imshow(my_output.reshape(28,28), cmap='gray')
            st.pyplot(fig3)