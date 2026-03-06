# file: models/ggnn.py

import tensorflow as tf
from tensorflow.keras import layers, Model


class MessagePassingLayer(layers.Layer):
    """
    Performs:
        h_out = W_out(h)
        aggregate = A @ h_out
        h_in = W_in(aggregate)
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.w_out = layers.Dense(hidden_size, use_bias=False)
        self.w_in = layers.Dense(hidden_size, use_bias=False)

    def call(self, h, adjacency):
        """
        Args:
            h: [batch, num_nodes, hidden]
            adjacency: [batch, num_nodes, num_nodes]

        Returns:
            messages: [batch, num_nodes, hidden]
        """

        # W_out(h)
        h_out = self.w_out(h)

        # A @ h_out
        aggregated = tf.matmul(adjacency, h_out)

        # W_in(aggregated)
        messages = self.w_in(aggregated)

        return messages


class GGNN(Model):
    """
    Gated Graph Neural Network (GGNN)

    Args:
        hidden_size: hidden dimension
        num_steps: message passing steps
        input_dim: default 2048
    """

    def __init__(self, hidden_size: int, num_steps: int, input_dim: int = 2048):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_steps = num_steps
        self.input_dim = input_dim

        # Input projection (image → hidden)
        self.input_projection = layers.Dense(hidden_size, activation="tanh")

        # Message passing
        self.message_passing = MessagePassingLayer(hidden_size)

        # GRU update
        self.gru_cell = layers.GRUCell(hidden_size)

    def call(self, images, adjacency, training=False):
        """
        Args:
            images: [batch, num_nodes, 2048]
            adjacency: [batch, num_nodes, num_nodes]

        Returns:
            final_state: [batch, num_nodes, hidden]
            initial_state: [batch, num_nodes, hidden]
        """

        # 1️⃣ Project image features to hidden space
        h = self.input_projection(images)
        initial_state = h

        # Node mask (disable isolated nodes)
        node_enabled = tf.reduce_sum(adjacency, axis=-1)
        node_enabled = tf.cast(node_enabled > 0, tf.float32)
        node_enabled = tf.expand_dims(node_enabled, -1)

        # 2️⃣ Iterative message passing
        for _ in range(self.num_steps):

            # Compute messages
            messages = self.message_passing(h, adjacency)

            # Flatten for GRUCell: [batch*num_nodes, hidden]
            batch_size = tf.shape(h)[0]
            num_nodes = tf.shape(h)[1]

            h_flat = tf.reshape(h, [-1, self.hidden_size])
            msg_flat = tf.reshape(messages, [-1, self.hidden_size])

            # GRU update
            updated_state, _ = self.gru_cell(msg_flat, [h_flat])

            # Restore shape
            h = tf.reshape(updated_state, [batch_size, num_nodes, self.hidden_size])

            # Apply node mask
            h = h * node_enabled

        return h, initial_state


# ----------------------------------------------------
# Example usage
# ----------------------------------------------------

if __name__ == "__main__":

    batch_size = 4
    num_nodes = 10
    hidden_size = 256
    num_steps = 3

    model = GGNN(hidden_size=hidden_size, num_steps=num_steps)

    images = tf.random.normal([batch_size, num_nodes, 2048])
    adjacency = tf.random.uniform(
        [batch_size, num_nodes, num_nodes],
        minval=0,
        maxval=2,
        dtype=tf.int32,
    )
    adjacency = tf.cast(adjacency, tf.float32)

    final_state, initial_state = model(images, adjacency)

    print("Initial state shape:", initial_state.shape)
    print("Final state shape:", final_state.shape)