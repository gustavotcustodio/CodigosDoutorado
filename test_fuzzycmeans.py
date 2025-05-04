from fcmeans import FCM
import dataset_loader
import numpy as np

X, y = dataset_loader.select_dataset_function("german_credit")()

fcm = FCM(n_clusters=11, m=2)
fcm.fit(X)
fcm_labels = fcm.predict(X)
breakpoint()
