from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Embedding, Flatten, Concatenate, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_ncf(num_users: int, num_items: int, emb_dim: int = 32, hidden: int = 64, dropout: float = 0.2) -> Model:
    user_in = Input(shape=(1,), name="user")
    item_in = Input(shape=(1,), name="item")

    user_emb = Embedding(input_dim=num_users, output_dim=emb_dim, name="user_emb")(user_in)
    item_emb = Embedding(input_dim=num_items, output_dim=emb_dim, name="item_emb")(item_in)

    u = Flatten()(user_emb)
    i = Flatten()(item_emb)

    x = Concatenate()([u, i])
    x = Dense(hidden, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(hidden // 2, activation="relu")(x)
    out = Dense(1, activation="linear")(x)  # predict rating

    model = Model([user_in, item_in], out)
    model.compile(optimizer=Adam(1e-3), loss="mse")
    return model
