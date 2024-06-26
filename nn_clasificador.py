import numpy as np
import tensorflow as tf

class ServiceRecommendationModel:
    """
    A class to encapsulate the service recommendation model.
    
    Attributes:
        num_features (int): Number of tax features.
        num_services (int): Number of services.
        num_periods (int): Number of periods in a year.
        num_users (int): Number of example users.
        model (tf.keras.Model): TensorFlow model for service recommendation.
    """

    def __init__(self, num_features, num_services, num_periods):
        """
        Initializes the ServiceRecommendationModel with specified parameters.
        
        Args:
            num_features (int): Number of tax features.
            num_services (int): Number of services.
            num_periods (int): Number of periods in a year.
        """
        self.num_features = num_features
        self.num_services = num_services
        self.num_periods = num_periods
        self.num_users = 1000  # Default number of example users

        # Generate random user data
        self.user_features = np.random.randint(0, 2, size=(self.num_users, self.num_features))
        self.user_services = np.random.randint(0, 2, size=(self.num_users, self.num_services))
        self.user_periods = np.random.randint(0, self.num_periods, size=self.num_users)

        # Define the neural network architecture
        self.model = self._build_model()

    def _build_model(self):
        """
        Builds a sequential Keras model for service recommendation.
        
        Returns:
            tf.keras.Model: Compiled Keras model.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(self.num_features + 1,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.num_services, activation='sigmoid')
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, epochs=10, batch_size=32):
        """
        Trains the defined Keras model using the generated user data.
        
        Args:
            epochs (int): Number of epochs for training.
            batch_size (int): Batch size for training.
        """
        self.model.fit(x=[np.concatenate((self.user_features, self.user_periods.reshape(-1, 1)), axis=1)],
                       y=self.user_services,
                       epochs=epochs,
                       batch_size=batch_size)

    def recommend_services(self, user_features, user_period, num_recommendations=5):
        """
        Recommends services for a new user based on their features and period.
        
        Args:
            user_features (numpy.ndarray): User's tax profile features.
            user_period (int): User's chosen period.
            num_recommendations (int): Number of top services to recommend.
        
        Returns:
            numpy.ndarray: Indices of recommended services.
        """
        user_data = np.concatenate((user_features, np.array([[user_period]])), axis=1)
        predictions = self.model.predict(user_data)
        top_indices = np.argsort(predictions[0])[::-1][:num_recommendations]
        return top_indices

if __name__ == '__main__':
    # Initialize the recommendation model
    recommendation_model = ServiceRecommendationModel(num_features=5, num_services=100, num_periods=56)

    # Generate data for a new user
    new_user_features = np.random.randint(0, 2, size=(1, recommendation_model.num_features))
    new_user_period = np.random.randint(0, recommendation_model.num_periods)

    # Get recommended services for the new user
    recommended_services = recommendation_model.recommend_services(new_user_features, new_user_period)

    # Print results
    print(f"""
    ******************
    New user: 
        Taxes = {new_user_features},
        Period = {new_user_period}
    ******************

    """)
    print("Recommended services for the new user:", recommended_services)
