from utility_model_file import MacularHole
from quiver_engine import server

nb_classes=2

m1=MacularHole(nb_rows=180,nb_cols=240,nb_channels=1)

model=m1.DeepConvo(nb_classes)

server.launch(model)