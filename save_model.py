import pickle


def save_model(clf_model, scaler, file, config):
    try:
        with open(file, 'wb') as pfile:
            pickle.dump({'classifier': clf_model,
                         'scaler': scaler,
                         'orientations': config['orientations'],
                         'pixels_per_cell': config['pixels_per_cell'],
                         'cells_per_block': config['cells_per_block'],
                         'visualize': config['visualize']
                         },

                        pfile, pickle.HIGHEST_PROTOCOL)
            print("Model saved to {}".format(file))

    except Exception as e:
        print("Failed to save the model at the destination file {}:{}".format(file, e))
        raise


def save_feature_data(file, feature):
    try:
        pickle.dump(feature, open(file, 'wb'), pickle.HIGHEST_PROTOCOL)
        print("Feature Data saved to {}".format(file))

    except Exception as e:
        print('Failed to save the model at the destination file {}:{}'.format(file, e))
        raise


def load_model(file):
    try:
        with open(file, 'rb') as pfile:
            clf = pickle.load(pfile)

        return clf
        print('trained model {} is loaded'.format(file))
    except Exception as e:
        print("Failed to load the model from the source {} file".format(file))
        raise
