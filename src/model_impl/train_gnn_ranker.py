import os
import json
import random
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from datetime import datetime

# =========================
# Configuration
# =========================

class Config:
    def __init__(self):
        self.batch_size = 16
        self.hidden_size = 64
        self.gnn_steps = 3
        self.learning_rate = 0.001
        self.epochs = 20
        self.beta = 0.0001
        self.feature_path = "/data/polyvore_image_vectors/polyvore_image_vectors/"
        self.train_json = "/data/train_no_dup_new_100.json"
        self.category_json = "/data/category_summarize_100.json"
        self.cid_map_json = "/data/cid2rcid_100.json"


# =========================
# GGNN Model (TF2)
# =========================

class MessagePassing(layers.Layer):
    def __init__(self, hidden_size):
        super().__init__()
        self.w_out = layers.Dense(hidden_size, use_bias=False)
        self.w_in = layers.Dense(hidden_size, use_bias=False)

    def call(self, h, adj):
        h_out = self.w_out(h)
        agg = tf.matmul(adj, h_out)
        return self.w_in(agg)


class GGNN(Model):
    def __init__(self, hidden_size, steps):
        super().__init__()
        self.hidden_size = hidden_size
        self.steps = steps
        self.input_proj = layers.Dense(hidden_size, activation="tanh")
        self.message = MessagePassing(hidden_size)
        self.gru = layers.GRUCell(hidden_size)

    def call(self, images, adj):
        h = self.input_proj(images)

        for _ in range(self.steps):
            msg = self.message(h, adj)
            B, N, H = msg.shape
            h_flat = tf.reshape(h, [-1, H])
            msg_flat = tf.reshape(msg, [-1, H])
            new_h, _ = self.gru(msg_flat, [h_flat])
            h = tf.reshape(new_h, [B, N, H])

        return h


class RankModel(Model):
    def __init__(self, hidden_size, steps):
        super().__init__()
        self.gnn = GGNN(hidden_size, steps)
        self.w_conf = layers.Dense(1, activation="sigmoid")
        self.w_score = layers.Dense(1)

    def graph_score(self, images, adj):
        h = self.gnn(images, adj)
        conf = self.w_conf(h)
        score = tf.nn.leaky_relu(self.w_score(h))
        node_score = conf * score
        return tf.reduce_sum(node_score, axis=1)

    def call(self, pos_img, pos_adj, neg_img, neg_adj):
        s_pos = self.graph_score(pos_img, pos_adj)
        s_neg = self.graph_score(neg_img, neg_adj)
        return s_pos, s_neg


# =========================
# Data Loader
# =========================

class DataLoader:
    def __init__(self, cfg):
        self.cfg = cfg
        self.category = json.load(open(cfg.category_json))
        self.cid_map = json.load(open(cfg.cid_map_json))
        self.train_data = json.load(open(cfg.train_json))
        self.num_category = len(self.category)

    def load_feature(self, sid, iid):
        path = os.path.join(self.cfg.feature_path, f"{sid}_{iid}.json")
        return np.array(json.load(open(path)), dtype=np.float32)

    def build_graph(self, cats):
        graph = np.zeros((self.num_category, self.num_category), dtype=np.float32)
        for a in cats:
            for b in cats:
                if a != b:
                    graph[a][b] = 1.
        return graph

    def sample_batch(self):
        batch = random.sample(self.train_data, self.cfg.batch_size)

        img_pos = np.zeros((self.cfg.batch_size, self.num_category, 2048), dtype=np.float32)
        img_neg = np.zeros_like(img_pos)
        g_pos = np.zeros((self.cfg.batch_size, self.num_category, self.num_category), dtype=np.float32)
        g_neg = np.zeros_like(g_pos)

        for i, outfit in enumerate(batch):
            ii = outfit["items_index"]
            ci = outfit["items_category"]
            sid = outfit["set_id"]

            replace_idx = random.randint(0, len(ii) - 1)

            pos_cats = []
            neg_cats = []

            for k in range(len(ii)):
                rcid = int(self.cid_map[str(ci[k])])
                feat = self.load_feature(sid, ii[k])

                if k == replace_idx:
                    img_pos[i][rcid] = feat
                    pos_cats.append(rcid)

                    rand_cat = random.choice(self.category)
                    rcid_neg = int(self.cid_map[str(rand_cat["id"])])
                    rand_item = random.choice(rand_cat["items"])
                    img_neg[i][rcid_neg] = np.array(
                        json.load(open(self.cfg.feature_path + rand_item + ".json")),
                        dtype=np.float32
                    )
                    neg_cats.append(rcid_neg)
                else:
                    img_pos[i][rcid] = feat
                    img_neg[i][rcid] = feat
                    pos_cats.append(rcid)
                    neg_cats.append(rcid)

            g_pos[i] = self.build_graph(pos_cats)
            g_neg[i] = self.build_graph(neg_cats)

        return img_pos, g_pos, img_neg, g_neg


# =========================
# Training Loop
# =========================

def train():
    cfg = Config()
    loader = DataLoader(cfg)

    model = RankModel(cfg.hidden_size, cfg.gnn_steps)
    optimizer = tf.keras.optimizers.Adam(cfg.learning_rate)

    for epoch in range(cfg.epochs):
        img_pos, g_pos, img_neg, g_neg = loader.sample_batch()

        with tf.GradientTape() as tape:
            s_pos, s_neg = model(img_pos, g_pos, img_neg, g_neg)
            loss = -tf.reduce_mean(tf.nn.sigmoid(s_pos - s_neg))

            l2 = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables])
            loss += cfg.beta * l2

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        print(f"[{datetime.now()}] Epoch {epoch} Loss {loss.numpy():.6f}")

    model.save_weights("ggnn_ranker.weights.h5")


if __name__ == "__main__":
    train()