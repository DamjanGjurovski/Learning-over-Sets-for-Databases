import pickle
from helper_methods import q_loss
from keras.models import load_model

'''Store the model as weights so you can see the true size'''
m_name = "SE-q_loss-l-e16-m16-d16-hashtags3scale-3_decayTrue_emb4_remove_F_min_boundary.h5"
location = "edbt_models/hashtags3/boundaries/"

nn_model = load_model(location + m_name, custom_objects={"q_loss": q_loss})
# nn_model = load_model(location + m_name)
print(nn_model.weights)

name_pickle = m_name + "_weights"
location_pickle = location + "weights_pickle/"
with open(location_pickle + name_pickle + ".pickle", 'wb') as f:
    pickle.dump(nn_model.weights,f)